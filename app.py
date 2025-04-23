import streamlit as st
import os
import base64
import tempfile
import time
import uuid
import io
import requests
import json
from PIL import Image
import fitz
from pdf2image import convert_from_path
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import CLIPModel, AutoProcessor
from qdrant_client import QdrantClient, models
import shutil

# --- Configuration ---

#POPPLER_PATH = r'C:\poppler-24.08.0\Library\bin'
# Set path for poppler dep
POPPLER_PATH = os.environ.get('POPPLER_PATH', '/usr/bin')

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CACHE_DIR = "./hf_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Streamlit App Setup ---
st.set_page_config(layout="wide")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

if "messages" not in st.session_state:
    st.session_state.messages = []
session_id = st.session_state.id

def reset_chat():
    # Store a reference to the current file if it exists
    current_file = None
    if "query_engine" in st.session_state:
        if hasattr(st.session_state.query_engine, 'uploaded_file'):
            current_file = st.session_state.query_engine.uploaded_file
    
    # Clear session state items
    st.session_state.messages = []
    if "query_engine" in st.session_state:
        del st.session_state.query_engine
    st.session_state.file_cache = {}
    
    # Clean up temp directory
    if os.path.exists("temp_images"):
        try:
            shutil.rmtree("temp_images")
        except Exception as e:
            st.warning(f"Could not remove temp_images directory: {e}")
    
    # Don't call st.rerun() when processing a new file - that's causing the loop
    # We'll only rerun if this is called from the reset button
    if current_file is None:
        st.rerun()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.getvalue()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- RAG Component Classes (Integrated) ---

class QueryEngine:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.processed_data = []
        self.embedded_data_clip = []
        self.clip_model = None
        self.clip_processor = None
        self.qdrant_client = None
        self.collection_name = f"clip_multimodal_pdf_rag_{session_id}" # Unique collection per session
        self.embedding_dimension_clip = None
        self.ollama_model_name = 'gemma3:latest'
        self.ollama_api_base = "http://ollama:11434"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self._process_pdf()
        self._load_embedding_model()
        self._generate_embeddings()
        self._setup_qdrant()
        self._ingest_data()

    def _process_pdf(self):
        st.write("Processing PDF...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, self.uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(self.uploaded_file.getvalue())

                pil_images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
                doc = fitz.open(file_path)

                for i, page_image in enumerate(tqdm(pil_images, desc="Extracting pages")):
                    page_text = doc[i].get_text("text") if i < len(doc) else ""
                    page_text = ' '.join(page_text.split())

                    buffered = io.BytesIO()
                    if page_image.mode == 'RGBA':
                        page_image = page_image.convert('RGB')
                    page_image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    self.processed_data.append({
                        "id": str(uuid.uuid4()),
                        "page_num": i + 1,
                        "text": page_text,
                        "image_pil": page_image,
                        "image_b64": img_base64
                    })
                doc.close()
            st.write(f"Successfully processed {len(self.processed_data)} pages/chunks.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()

    def _load_embedding_model(self):
        st.write(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        try:
            # Check if the cache directory exists, otherwise create it
            if not os.path.exists(CACHE_DIR):
                st.write(f"Cache directory {CACHE_DIR} does not exist. Creating it...")
                os.makedirs(CACHE_DIR, exist_ok=True)
                st.write(f"Cache directory created successfully.")
            
            st.write(f"Checking for model in cache directory: {CACHE_DIR}")
            # Load model from cache if available, otherwise download it
            self.clip_model = CLIPModel.from_pretrained(
                CLIP_MODEL_NAME,
                cache_dir=CACHE_DIR
            ).to(self.DEVICE).eval()
            
            st.write(f"Loading processor from {CLIP_MODEL_NAME}")
            self.clip_processor = AutoProcessor.from_pretrained(
                CLIP_MODEL_NAME,
                cache_dir=CACHE_DIR
            )
            
            # Determine embedding dimension
            if hasattr(self.clip_model.config, 'projection_dim'):
                self.embedding_dimension_clip = self.clip_model.config.projection_dim
            elif hasattr(self.clip_model.config, 'hidden_size'):
                self.embedding_dimension_clip = self.clip_model.config.hidden_size
            else:
                st.warning("Could not automatically determine embedding dimension from model config.")
                # Fallback dimension for CLIP-base-patch32
                self.embedding_dimension_clip = 512  # Assume 512 for this specific model

            st.write(f"CLIP Embedding dimension: {self.embedding_dimension_clip}")
            st.write("CLIP model and processor loaded successfully.")
        except Exception as e:
            st.error(f"Error loading CLIP model/processor: {e}")
            st.stop()

    def _generate_embeddings(self):
        if not self.clip_model or not self.clip_processor:
            st.warning("CLIP model or processor not loaded. Skipping embedding generation.")
            return

        st.write(f"Generating CLIP IMAGE embeddings for {len(self.processed_data)} items...")
        for chunk in tqdm(self.processed_data, desc="Generating Image Embeddings"):
            try:
                image_pil = chunk['image_pil']
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')

                inputs = self.clip_processor(images=image_pil, return_tensors="pt", padding=True).to(self.DEVICE)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                image_embedding_vector = image_features[0].cpu().float().numpy().tolist()

                if image_embedding_vector:
                    chunk['embedding'] = image_embedding_vector
                    self.embedded_data_clip.append(chunk)
                else:
                    st.warning(f"Skipping chunk on page {chunk['page_num']} due to image embedding error.")
            except Exception as e:
                 st.warning(f"Error generating embedding for page {chunk['page_num']}: {e}")


        if self.embedded_data_clip:
             if self.embedding_dimension_clip and len(self.embedded_data_clip[0]['embedding']) != self.embedding_dimension_clip:
                 st.warning(f"Actual embedding dimension {len(self.embedded_data_clip[0]['embedding'])} differs from config dimension {self.embedding_dimension_clip}. Updating dimension.")
                 self.embedding_dimension_clip = len(self.embedded_data_clip[0]['embedding'])

             st.write(f"\nSuccessfully generated {len(self.embedded_data_clip)} CLIP image embeddings.")
             st.write(f"Using embedding dimension: {self.embedding_dimension_clip}")
        else:
             st.warning("\nNo CLIP image embeddings were generated.")
             self.embedding_dimension_clip = None


    def _setup_qdrant(self):
        if not self.embedding_dimension_clip:
            st.warning("Embedding dimension not determined. Skipping Qdrant setup.")
            return

        st.write("Attempting to connect to Qdrant on localhost (port 6334) or Docker alias (port 6334)...")

        try:
            # Try connecting to local Docker or local host on gRPC port
            self.qdrant_client = QdrantClient(host="qdrant", port=6333, timeout=5)
            self.qdrant_client.get_collections() # Test connection
            st.write("Successfully connected to Qdrant.")
        except Exception:
             try:
                self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=5)
                self.qdrant_client.get_collections() # Test connection
                st.write("Successfully connected to Qdrant on localhost.")
             except Exception as e:
                st.error(f"Error connecting to Qdrant: {e}. Ensure it's running locally.")
                self.qdrant_client = None
                return # Exit if connection fails


        if self.qdrant_client:
            try:
                st.write(f"Checking/Creating Qdrant collection: '{self.collection_name}'")
                collections_response = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections_response.collections]

                if self.collection_name in collection_names:
                    st.write(f"Collection '{self.collection_name}' exists. Deleting and recreating.")
                    self.qdrant_client.delete_collection(collection_name=self.collection_name)
                    time.sleep(1)

                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension_clip,
                        distance=models.Distance.COSINE
                    )
                )
                st.write(f"Collection '{self.collection_name}' created successfully with dimension {self.embedding_dimension_clip}.")

            except Exception as e:
                st.error(f"Error during Qdrant collection setup: {e}")
                self.qdrant_client = None # Invalidate client if collection setup fails

    def _ingest_data(self):
        BATCH_SIZE = 64
        if not self.qdrant_client or not self.embedded_data_clip:
            st.warning("Skipping ingestion: Qdrant client not connected or no data with CLIP embeddings available.")
            return

        st.write(f"Ingesting {len(self.embedded_data_clip)} data points into Qdrant...")
        total_ingested = 0
        num_batches = (len(self.embedded_data_clip) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(0, len(self.embedded_data_clip), BATCH_SIZE), desc="Ingesting Batches", total=num_batches):
            batch = self.embedded_data_clip[i : i + BATCH_SIZE]
            points_to_upsert = []
            for item in batch:
                 if 'embedding' in item and isinstance(item['embedding'], list):
                    points_to_upsert.append(
                         models.PointStruct(
                             id=item['id'],
                             vector=item['embedding'],
                             payload={
                                 "text": item['text'],
                                 "page_num": item['page_num'],
                                 "image_b64": item['image_b64']
                             }
                         )
                    )
                 else:
                     st.warning(f"Skipping ingestion for item ID {item.get('id', 'N/A')} due to missing/invalid embedding.")

            if points_to_upsert:
                try:
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=points_to_upsert, wait=True) # Use wait=True for Streamlit
                    total_ingested += len(points_to_upsert)
                except Exception as e:
                    st.error(f"Error upserting batch to Qdrant: {e}")
                    # Decide if you want to stop or continue on error

        st.write(f"\nIngestion complete. Total points ingested: {total_ingested}")
        try:
            count = self.qdrant_client.count(collection_name=self.collection_name, exact=True)
            st.write(f"Verification: Qdrant reports {count.count} points in the collection.")
        except Exception as e:
            st.warning(f"Could not verify count in Qdrant: {e}")

    def _get_clip_text_embedding(self, text_query):
        if not self.clip_model or not self.clip_processor:
            st.warning("CLIP model or processor not loaded. Cannot generate text embedding.")
            return None
        try:
            inputs = self.clip_processor(text=[text_query], return_tensors="pt", padding=True).to(self.DEVICE)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features[0].cpu().float().numpy().tolist()
        except Exception as e:
            st.error(f"Error generating text query embedding: {e}")
            return None

    def _retrieve_context(self, query_embedding, top_k=2):
        if not self.qdrant_client or not query_embedding:
            st.warning("Qdrant client not initialized or query embedding is missing.")
            return []

        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            return search_result
        except Exception as e:
            st.error(f"Error during Qdrant search: {e}")
            return []

    def _prepare_and_generate(self, query, retrieved_results):
        if not retrieved_results:
            yield "No relevant information found to generate a response."
            return

        # Create a directory for temporary images if it doesn't exist
        os.makedirs("temp_images", exist_ok=True)
        
        # Build a prompt with context from retrieved results
        prompt = f"I have a question about a document: {query}\n\nHere are relevant parts of the document to help you answer:\n\n"
        
        # List to store base64-encoded images
        base64_images = []
        
        for i, result in enumerate(retrieved_results):
            if not result.payload:
                continue
                
            context_payload = result.payload
            context_text_content = context_payload.get('text', '')
            context_page = context_payload.get('page_num', 'N/A')
            relevance_score = result.score
            
            prompt += f"--- Document Page {context_page} (Relevance Score: {relevance_score:.4f}) ---\n"
            if context_text_content:
                prompt += f"Text: {context_text_content}\n\n"
            
            # For each result with an image, add its base64 encoding to our list
            if 'image_b64' in context_payload and context_payload['image_b64']:
                # Save image to temp file to make debugging easier if needed
                img_data = base64.b64decode(context_payload['image_b64'])
                img_path = f"temp_images/page_{context_page}_{i}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_data)
                    
                # Add the base64 string directly to our list (not as data URL)
                base64_images.append(context_payload['image_b64'])
        
        # Finalize the prompt with the actual question
        prompt += f"\nPlease answer my question using both the text and visual information from the document."
        
        # Generate the answer using Ollama Generate API (for multimodal support)
        try:
            # For Ollama's generate API (not chat), the format is different
            generate_payload = {
                "model": self.ollama_model_name,
                "prompt": prompt,
                "stream": True
            }
            
            # Add images if we have any (using the correct format for generate API)
            if base64_images:
                generate_payload["images"] = base64_images
                
            response = requests.post(
                f"{self.ollama_api_base}/api/generate",
                json=generate_payload,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                yield json_response['response']

                            if json_response.get('done', False):
                                break
                        except json.JSONDecodeError:
                            pass
                yield "\n"
            else:
                yield f"Error: Ollama API returned status {response.status_code} - {response.text}"

        except requests.exceptions.ConnectionError:
            yield f"Error: Could not connect to Ollama at {self.ollama_api_base}. Is Ollama running?"
        except Exception as e:
            yield f"Error during generation: {e}"

    def query(self, query_text):
        # Clear temp_images directory at the beginning of each query
        if os.path.exists("temp_images"):
            try:
                for file in os.listdir("temp_images"):
                    file_path = os.path.join("temp_images", file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            except Exception as e:
                print(f"Error clearing temp_images: {e}")
        
        # Recreate the directory if it doesn't exist
        os.makedirs("temp_images", exist_ok=True)
        
        # Embed the query text using CLIP
        query_embedding = self._get_clip_text_embedding(query_text)

        if not query_embedding:
            return iter(["Error: Could not generate query embedding."]) # Return an iterator

        # Retrieve relevant chunks from Qdrant (based on image embeddings)
        retrieved_results = self._retrieve_context(query_embedding)

        # Prepare context and generate response using Ollama (streaming)
        return self._prepare_and_generate(query_text, retrieved_results)

# --- Streamlit App Layout ---

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    # Multimodal RAG powered by Gemma 3 and Ollama
    """)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        # Check if we need to process a new file or if it's already loaded
        file_key = uploaded_file.name + str(hash(uploaded_file.getvalue()))
        
        if file_key not in st.session_state.file_cache:
            # Clear previous session data if a different file was uploaded
            if "query_engine" in st.session_state:
                # Clear without forcing a rerun
                st.session_state.messages = []
                if "query_engine" in st.session_state:
                    del st.session_state.query_engine
                st.session_state.file_cache = {}
                if os.path.exists("temp_images"):
                    try:
                        shutil.rmtree("temp_images")
                    except Exception as e:
                        st.warning(f"Could not remove temp_images directory: {e}")
                
                st.info("Processing new document.")
            
            # Process the new file
            with st.spinner("Processing your document. This may take a moment..."):
                st.session_state.query_engine = QueryEngine(uploaded_file)
                st.session_state.file_cache[file_key] = True
            
            st.success("Document processed and ready to chat!")
        elif "query_engine" not in st.session_state:
            # Case where the file is in cache but engine not loaded (after page refresh)
            with st.spinner("Processing your document. This may take a moment..."):
                st.session_state.query_engine = QueryEngine(uploaded_file)
            
            st.success("Document processed and ready to chat!")
        else:
            st.info("Document already loaded. Ready to chat!")
        
        # Always show PDF preview when a file is uploaded
        display_pdf(uploaded_file)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if "query_engine" in st.session_state and st.session_state.query_engine:
            try:
                for chunk in st.session_state.query_engine.query(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌") # Add blinking cursor while streaming
                message_placeholder.markdown(full_response) # Display final response
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                message_placeholder.markdown(error_message)
                full_response = error_message
        else:
            full_response = "Please upload a PDF document in the sidebar to begin."
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})