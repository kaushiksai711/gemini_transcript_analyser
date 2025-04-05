"""
Gemini Transcript Analysis MVP with Enhanced Batch Processing

This is the main entry point that integrates all components:
1. Streamlit UI with advanced visualization
2. Batch processing with dependency tracking
3. Long context handling for large transcripts
4. Context caching for token optimization

Run with: streamlit run main.py
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("cache", exist_ok=True)
os.makedirs("samples", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# Import Streamlit and related modules
import streamlit as st

# Import Gemini MVP modules
from gemini_mvp.transcript_processor import TranscriptProcessor
from gemini_mvp.batch_processor import BatchProcessor
from gemini_mvp.long_context_handler import LongContextHandler
from gemini_mvp.context_cache import ContextCache
from gemini_mvp.cache_manager import CacheManager
from gemini_mvp.api_client import GeminiClient
from gemini_mvp.batch_ui import render_batch_ui

# Import original Streamlit UI functionality
# This will import the main app layout, tabs, and basic functionality
# The batch_ui module extends it with advanced batch processing features
from streamlit_ui import process_transcript_pipeline

# Set page config
st.set_page_config(
    page_title="Gemini MVP - Transcript Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-badge {
        background-color: #3498db;
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.8rem;
        margin-left: 5px;
    }
    
    .new-feature {
        background-color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>Gemini Transcript Analysis MVP</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Process video transcripts and answer questions with advanced context management</p>", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if "processed_transcript" not in st.session_state:
    st.session_state.processed_transcript = False

if "processor" not in st.session_state:
    st.session_state.processor = None

# Create main tabs with enhanced batch tab
tabs = st.tabs([
    "📝 Setup", 
    "📋 Process Transcript", 
    "🧩 Chunk Analysis", 
    "❓ Single Query", 
    "🔄 Batch Analysis",
    "🚀 Enhanced Batch Processing",
    "📊 Stats"
])

# Setup tab
with tabs[0]:
    st.markdown("<h2 class='section-header'>Configuration</h2>", unsafe_allow_html=True)
    
    # API Key configuration
    api_key_col, info_col = st.columns([2, 1])
    
    with api_key_col:
        api_key = st.text_input(
            "Google API Key", 
            type="password", 
            placeholder="Enter your Google API key here",
            value=os.environ.get("GOOGLE_API_KEY", "")
        )
        
        if st.button("Save API Key", use_container_width=True):
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.session_state.api_key_set = True
                st.success("API key saved successfully!")
            else:
                st.error("Please enter a valid API key")
    
    with info_col:
        st.info("Your API key is required to use the Gemini models. It will be stored in your environment variables.")
    
    # Model selection
    model_col, model_info_col = st.columns([2, 1])
    
    with model_col:
        model_options = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-1.0-pro-vision"
        ]
        
        selected_model = st.selectbox(
            "Select Gemini Model",
            options=model_options,
            index=0,
            help="Choose which Gemini model to use for processing"
        )
        
        st.session_state.model = selected_model
    
    with model_info_col:
        st.info("Different models have different capabilities and token limits. gemini-1.5-pro has a 1M token context window.")

# Reuse original Streamlit UI functionality for tabs 1-4 and 6
# We'll implement our enhanced batch processing in tab 5

# Process Transcript tab (tab 1)
with tabs[1]:
    st.markdown("<h2 class='section-header'>Process Your Transcript</h2>", unsafe_allow_html=True)
    
    # Transcript upload interface
    upload_col, info_col = st.columns([2, 1])
    
    with upload_col:
        uploaded_file = st.file_uploader("Upload Transcript", type=["txt"], help="Upload a transcript file in text format")
        sample_checkbox = st.checkbox("Use sample transcript instead", value=False)
        
    with info_col:
        st.info("Upload a transcript file or use our sample transcript to get started.")
    
    if st.session_state.api_key_set:
        if uploaded_file or sample_checkbox:
            process_button = st.button("Process Transcript", type="primary", use_container_width=True)
            
            if process_button:
                # Process the transcript
                try:
                    if uploaded_file:
                        # Save uploaded file
                        transcript_path = Path("uploaded_transcript.txt")
                        with open(transcript_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                    else:
                        # Use sample transcript
                        transcript_path = Path("samples/sample_transcript.txt")
                        
                        # Create sample if it doesn't exist
                        if not transcript_path.exists():
                            from batch_demo import load_or_create_sample_transcript
                            sample_text = load_or_create_sample_transcript()
                            transcript_path = Path("samples/sample_transcript.txt")
                    
                    # Process the transcript
                    process_transcript_pipeline(
                        transcript_path=transcript_path,
                        config={
                            "api_key": os.environ.get("GOOGLE_API_KEY"),
                            "model_name": st.session_state.model
                        }
                    )
                    
                    # Initialize enhanced components if needed
                    if not hasattr(st.session_state, "long_context_handler") and hasattr(st.session_state, "processor"):
                        processor = st.session_state.processor
                        
                        # Create cache manager if not exists
                        if not hasattr(st.session_state, "cache_manager"):
                            st.session_state.cache_manager = CacheManager({
                                "disk_cache_dir": "cache",
                                "memory_cache_size": 100
                            })
                        
                        # Initialize long context handler
                        st.session_state.long_context_handler = LongContextHandler(
                            chunk_manager=processor.chunk_manager,
                            cache_manager=st.session_state.cache_manager
                        )
                        
                        # Initialize context cache
                        st.session_state.context_cache = ContextCache(
                            cache_manager=st.session_state.cache_manager
                        )
                        
                        # Initialize batch processor
                        st.session_state.batch_processor = BatchProcessor(
                            api_client=processor.api_client,
                            query_processor=processor.query_processor,
                            context_manager=processor.context_manager,
                            cache_manager=st.session_state.cache_manager
                        )
                    
                except Exception as e:
                    st.error(f"Error processing transcript: {e}")
                    st.exception(e)
    else:
        st.warning("Please set your API key in the Setup tab first")

# Enhanced Batch Processing tab (tab 5)
with tabs[5]:
    if st.session_state.processed_transcript and st.session_state.processor:
        # Render enhanced batch UI
        render_batch_ui(
            processor=st.session_state.processor,
            api_client=st.session_state.processor.api_client,
            context_manager=st.session_state.processor.context_manager,
            cache_manager=st.session_state.cache_manager if hasattr(st.session_state, "cache_manager") else None
        )
    else:
        st.info("Please process a transcript in the Process Transcript tab first")

# Add a footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Gemini Transcript Analysis MVP with Enhanced Batch Processing</p>
    <p style="font-size: 0.8rem">Powered by Google Gemini 1.5 Pro • Optimized for large context windows</p>
</div>
""", unsafe_allow_html=True)

def main():
    """Main entrypoint function (not used with Streamlit)"""
    print("Please run this application with: streamlit run main.py")


if __name__ == "__main__":
    # For direct Python execution (will show message)
    if not 'streamlit' in sys.modules:
        main()
