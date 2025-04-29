import streamlit as st
import os
import base64
import uuid
import shutil
import sys

#Make sure the current directory is in the path so we can import the rag_engine module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import QueryEngine

# Configuration
#POPPLER_PATH = r'C:\poppler-24.08.0\Library\bin'
POPPLER_PATH = os.environ.get('POPPLER_PATH', '/usr/bin') #Use in production

# Streamlit App Layout
st.set_page_config(layout="wide")

# Initialize session state
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
    
    # Only rerun if this is called from the reset button
    if current_file is None:
        st.rerun()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.getvalue()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

def streamlit_progress_callback(msg):
    """Callback function to display progress in Streamlit"""
    st.write(msg)

# App Header
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    # Multimodal RAG powered by Gemma 3 and Ollama
    """)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Sidebar
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
                try:
                    st.session_state.query_engine = QueryEngine(
                        uploaded_file, 
                        session_id,
                        progress_callback=streamlit_progress_callback,
                        poppler_path=POPPLER_PATH
                    )
                    st.session_state.file_cache[file_key] = True
                    st.success("Document processed and ready to chat!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        elif "query_engine" not in st.session_state:
            # Case where the file is in cache but engine not loaded (after page refresh)
            with st.spinner("Processing your document. This may take a moment..."):
                try:
                    st.session_state.query_engine = QueryEngine(
                        uploaded_file, 
                        session_id,
                        progress_callback=streamlit_progress_callback,
                        poppler_path=POPPLER_PATH
                    )
                    st.success("Document processed and ready to chat!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
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
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                message_placeholder.markdown(error_message)
                full_response = error_message
        else:
            full_response = "Please upload a PDF document in the sidebar to begin."
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})