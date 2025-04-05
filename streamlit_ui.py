"""
Streamlit UI for Gemini Transcript Analysis MVP.

This module provides a visually appealing and comprehensive Streamlit-based UI for interacting with
the Gemini MVP for transcript analysis, showing the entire workflow from transcript processing
to chunking to final answers.
"""

import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from io import StringIO

# Fix PyTorch module path issue with Streamlit
import streamlit as st

from gemini_mvp import query_processor
os.environ["STREAMLIT_WORKAROUNDS_TORCH"] = "1"

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Gemini MVP modules
from gemini_mvp.api_client import GeminiClient
from gemini_mvp.cache_manager import CacheManager
from gemini_mvp.context_manager import ContextManager
from gemini_mvp.dependency_tracker import DependencyTracker
from gemini_mvp.transcript_processor import TranscriptProcessor
from gemini_mvp.query_processor import QueryProcessor

# Ensure necessary directories exist
os.makedirs("cache", exist_ok=True)
os.makedirs("samples", exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Gemini MVP - Transcript Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if "processed_transcript" not in st.session_state:
    st.session_state.processed_transcript = False

if "context_weights" not in st.session_state:
    st.session_state.context_weights = {
        "recency": 0.3,
        "relevance": 0.6, 
        "entity": 0.4,
        "topic": 0.3,
        "concept": 0.2
    }

if "visualize_chunks" not in st.session_state:
    st.session_state.visualize_chunks = True
    
if "chunk_vis_data" not in st.session_state:
    st.session_state.chunk_vis_data = None
    
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
    
if "cache_misses" not in st.session_state:
    st.session_state.cache_misses = 0
    
if "total_saved_tokens" not in st.session_state:
    st.session_state.total_saved_tokens = 0
    
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

if "performance_data" not in st.session_state:
    st.session_state.performance_data = {
        "query_times": [],
        "context_selection_times": [],
        "api_response_times": []
    }

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
    
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: black;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .answer-card {
        background-color: #f1f8e9;
        border-left: 5px solid #8bc34a;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .confidence-score {
        font-size: 0.9rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    /* Fix for white text on light background */
    input, textarea, div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    /* Make placeholder text darker */
    ::placeholder {
        color: #666666 !important;
        opacity: 0.8 !important;
    }
    
    /* Add border to input fields for better visibility */
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        border: 1px solid #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for processing
def process_transcript_pipeline(transcript_path: Path, config: Dict[str, Any]) -> None:
    """Process a transcript through the entire pipeline with visual feedback"""
    
    # Create a progress bar
    progress_placeholder = st.empty()
    progress_placeholder.progress(0)
    
    status_text = st.empty()
    status_text.text("Initializing components...")
    
    # Initialize components
    cache_manager = CacheManager({
        "disk_cache_dir": "cache",
        "memory_cache_size": 100
    })
    
    # Configure API client with cache
    api_client = GeminiClient(
        api_key=config["api_key"],
        config={"model": config["model_name"]}
    )
    
    # Attach cache manager to API client
    api_client.cache_manager = cache_manager
    
    # Update progress
    progress_placeholder.progress(0.1)
    status_text.text("Reading transcript...")
    
    # Read transcript
    with open(transcript_path, "r") as f:
        transcript_text = f.read()
    
    # Update progress
    progress_placeholder.progress(0.2)
    status_text.text("Creating transcript processor...")
    
    # Create processor
    processor = TranscriptProcessor(
        transcript_text=transcript_text,
        config_path=config.get("config_path", "config.yaml"),
        api_key=config["api_key"],
        cache_dir=config.get("cache_dir", "cache"),
        verbose=config.get("verbose", False)
    )
    
    # Update progress
    progress_placeholder.progress(0.3)
    status_text.text("Processing transcript into chunks...")
    
    # No need to call process_transcript separately as it's done in the constructor
    # Just verify that transcript was loaded successfully
    if not processor.transcript_loaded:
        raise Exception("Failed to process transcript")
    
    # Update progress
    progress_placeholder.progress(0.5)
    status_text.text("Entities and topics are already extracted during chunk processing...")
    
    # Update progress
    progress_placeholder.progress(0.7)
    status_text.text("Creating context manager...")
    
    # Create context manager
    context_manager = ContextManager(
        config={
            "weights": config["context_weights"],
            "max_chunks": config["max_context_chunks"],
            "history_limit": 10
        },
        cache_manager=cache_manager
    )
    
    # Note: We don't need to explicitly set chunks here.
    # The context_manager.get_context method will use the chunks 
    # provided in the query_info dictionary when called.
    
    # Update progress
    progress_placeholder.progress(0.8)
    status_text.text("Initializing dependency tracker...")
    
    # Create dependency tracker
    dependency_tracker = DependencyTracker()
    
    # Update progress
    progress_placeholder.progress(0.9)
    status_text.text("Preparing visualization data...")
    
    # Prepare visualization data
    chunk_vis_data = prepare_chunk_visualization_data(processor.chunk_manager.chunks)
    
    # Update progress
    progress_placeholder.progress(1.0)
    status_text.text("Transcript processing complete!")
    
    # Store components in session state
    st.session_state.processor = processor
    st.session_state.context_manager = context_manager
    st.session_state.api_client = api_client
    st.session_state.dependency_tracker = dependency_tracker
    st.session_state.processed_transcript = True
    st.session_state.chunk_vis_data = chunk_vis_data
    
    # Clear progress indicators
    time.sleep(0.5)
    progress_placeholder.empty()
    status_text.empty()

def prepare_chunk_visualization_data(chunks) -> pd.DataFrame:
    """Prepare data for chunk visualization"""
    data = []
    
    for chunk in chunks:
        chunk_data = {
            "Chunk ID": chunk.chunk_id,
            "Length": len(chunk.text),
            "Timestamp (s)": getattr(chunk, "start_time", 0) if hasattr(chunk, "start_time") else 0
        }
        data.append(chunk_data)
    
    return pd.DataFrame(data)

def process_single_query(query, processor, api_client, max_chunks=8):
    """Process a single query and return the result"""
    # Get context manager from session state
    context_manager = st.session_state.context_manager
    print(context_manager,'asdadad')
    # Update max chunks
    context_manager.max_chunks = max_chunks
    config = {
                        "chunk_size": st.session_state.get("chunk_size", 300),
                        "chunk_overlap": st.session_state.get("chunk_overlap", 50),
                        "max_context_chunks": st.session_state.get("max_context_chunks", 8),
                        "use_semantic_chunking": st.session_state.get("semantic_chunking", True),
                        "api_key": os.environ.get("GEMINI_API_KEY"),
                        "model_name": "gemini-1.5-pro",
                        "context_weights": st.session_state.context_weights
                    }
    # Start timing
    start_time = time.time()
    
    query_processor = QueryProcessor(
        config=config,
        context_manager=context_manager
    )
    query_info = query_processor.process_query(query)
    
    # Add available chunks to query_info - this is the key fix
    if hasattr(processor, 'chunk_manager') and processor.chunk_manager.chunks:
        query_info["available_chunks"] = processor.chunk_manager.chunks
    
    #print(query_info,'query_info')
    # Get context for query
    selected_chunks = context_manager.get_context(query, query_info)
    #print(selected_chunks,'adasdsada')
    # Record context selection time
    context_time = time.time() - start_time
    st.session_state.performance_data["context_selection_times"].append(context_time)
    
    # Create a dictionary with context information for compatibility
    context_result = {
        "selected_chunks": selected_chunks,
        "query_complexity": query_processor.analyze_query_complexity(query) if hasattr(query_processor, 'analyze_query_complexity') else 0.5,
        "weights_used": context_manager.config.get("context_weights", {}),
        "scored_chunks": [(chunk, 1.0) for chunk in selected_chunks]  # Default score of 1.0 for each chunk
    }
    
    # Get chunks and their scores
    context_text = "\n\n".join([chunk.text for chunk in selected_chunks]) if selected_chunks else ""
    
    # Generate answer
    api_start_time = time.time()
    result = api_client.generate_response(query, selected_chunks, query_info)
    api_time = time.time() - api_start_time
    
    # Record API response time
    st.session_state.performance_data["api_response_times"].append(api_time)
    
    # Record total query time
    total_time = time.time() - start_time
    st.session_state.performance_data["query_times"].append(total_time)
    
    # Add query complexity and context info
    result["query_complexity"] = context_result.get("query_complexity", 0.5)
    result["weights_used"] = context_result.get("weights_used", {})
    
    # Add context chunks for debugging
    result["context_chunks"] = [
        {"chunk_id": chunk.chunk_id, "score": score, "text": chunk.text}
        for chunk, score in context_result.get("scored_chunks", [])[:max_chunks]
    ]
    
    # Update cache stats if available
    if hasattr(api_client, "get_stats"):
        stats = api_client.get_stats()
        if stats:
            st.session_state.api_calls = stats.get("api_calls", 0)
            st.session_state.cache_hits = stats.get("hits", 0)
            st.session_state.cache_misses = stats.get("misses", 0)
            st.session_state.total_saved_tokens = stats.get("total_saved_tokens", 0)
    
    print(result,'asassss')
    return result

def process_question_batch(
    questions, 
    processor, 
    api_client,
    batch_size=5,
    max_chunks=8,
    use_dependencies=True,
    progress_callback=None
):
    """Process a batch of questions with optional dependency tracking"""
    
    # Get components from session state
    context_manager = st.session_state.context_manager
    dependency_tracker = st.session_state.dependency_tracker if use_dependencies else None
    
    # Update max chunks
    context_manager.max_chunks = max_chunks
    config = {
                        "chunk_size": st.session_state.get("chunk_size", 300),
                        "chunk_overlap": st.session_state.get("chunk_overlap", 50),
                        "max_context_chunks": st.session_state.get("max_context_chunks", 8),
                        "use_semantic_chunking": st.session_state.get("semantic_chunking", True),
                        "api_key": os.environ.get("GEMINI_API_KEY"),
                        "model_name": "gemini-1.5-pro",
                        "context_weights": st.session_state.context_weights
                    }
    # Start timing
    start_time = time.time()
    
    query_processor = QueryProcessor(
        config=config,
        context_manager=context_manager
    )
    # Prepare context provider function
    def context_provider(query):
        context_start = time.time()
        query_info = query_processor.process_query(query)
    
        # Add available chunks to query_info - this is the key fix
        if hasattr(processor, 'chunk_manager') and processor.chunk_manager.chunks:
            query_info["available_chunks"] = processor.chunk_manager.chunks
        
        #print(query_info,'query_info')
        # Get context for query
        selected_chunks = context_manager.get_context(query, query_info)
        #print(selected_chunks,'adasdsada')
        
        # Record context selection time
        context_time = time.time() - context_start
        st.session_state.performance_data["context_selection_times"].append(context_time)
        
        #selected_chunks = context_result["selected_chunks"]
        context_text = "\n\n".join([chunk.text for chunk in selected_chunks])
        return context_text
    context_provider_list = []
    for question in questions:
        context_provider_list.append(context_provider(question))
    # Process questions in batch with dependency tracking
    dependencies = [] if use_dependencies else None
    results = api_client.process_batch(
        questions, 
        context_provider_list, 
        dependencies=dependencies, 
        #batch_size=batch_size,
        #progress_callback=progress_callback
    )
    
    # Record total batch time (average per question)
    total_time = (time.time() - start_time) / len(questions)
    st.session_state.performance_data["query_times"].extend([total_time] * len(questions))
    
    # Update cache stats if available
    if hasattr(api_client, "get_stats"):
        stats = api_client.get_stats()
        if stats:
            st.session_state.api_calls = stats.get("api_calls", 0)
            st.session_state.cache_hits = stats.get("hits", 0)
            st.session_state.cache_misses = stats.get("misses", 0)
            st.session_state.total_saved_tokens = stats.get("total_saved_tokens", 0)
    
    return results

# Create a sample transcript file if it doesn't exist
sample_transcript_path = Path("samples/sample_transcript.txt")
if not sample_transcript_path.exists():
    sample_transcript = """[00:00:00] Speaker 1: Welcome to our discussion on artificial intelligence and its impact on society.
[00:00:10] Speaker 1: Today we'll be covering recent advances in large language models, their applications, and ethical considerations.
[00:01:05] Speaker 2: Thank you for having me. I'd like to start by discussing how these models work at a high level.
[00:01:30] Speaker 2: Language models like GPT-4, Gemini, and Claude are trained on vast amounts of text data to predict the next word in a sequence.
[00:02:15] Speaker 1: And how has this technology evolved in recent years?
[00:02:30] Speaker 2: We've seen exponential growth in model size and capabilities since 2018, with models now able to perform complex reasoning, coding, and creative tasks.
[00:03:45] Speaker 2: For example, these models can now write coherent essays, debug complex code, and even pass bar exams.
[00:04:20] Speaker 1: What about the ethical concerns around these technologies?
[00:04:35] Speaker 2: Great question. There are several key concerns including bias in training data, potential for misinformation, privacy issues, and economic impacts.
[00:05:50] Speaker 2: For instance, if models are trained on biased data, they may perpetuate or amplify those biases in their outputs.
[00:06:30] Speaker 1: And how might we address these concerns?
[00:06:45] Speaker 2: It requires a multi-faceted approach: diverse training data, robust evaluation frameworks, transparency in development, and appropriate regulations.
[00:08:10] Speaker 2: Many organizations are now implementing ethical guidelines and responsible AI practices.
[00:09:00] Speaker 1: What applications do you see as most promising in the near term?
[00:09:15] Speaker 2: Healthcare shows particular promise, with AI assisting in diagnosis, drug discovery, and personalized medicine.
[00:10:30] Speaker 2: Education is another area where AI can provide personalized learning experiences at scale.
[00:11:45] Speaker 1: And finally, where do you see this technology heading in the next five years?
[00:12:00] Speaker 2: We'll likely see more multimodal models that understand and generate text, images, audio, and video seamlessly.
[00:13:20] Speaker 2: I also expect more specialized models for particular domains and further integration into everyday tools and workflows.
[00:14:30] Speaker 1: Thank you for these insights. It's clear we're just at the beginning of this technological revolution.
[00:14:45] Speaker 2: Indeed, and how we guide it will determine its ultimate impact on society. Thank you for having me.
"""
    os.makedirs("samples", exist_ok=True)
    with open(sample_transcript_path, "w") as f:
        f.write(sample_transcript)

# Create a sample questions file if it doesn't exist
sample_questions_path = Path("samples/sample_questions.csv")
if not sample_questions_path.exists():
    sample_questions = """question
What are the main topics discussed in this transcript?
Who are the speakers in this discussion?
What ethical concerns were raised about AI?
How have language models evolved according to Speaker 2?
What applications of AI were mentioned as promising?
What is predicted for AI technology in the next five years?
How do language models like GPT-4 and Gemini work?
What approaches were suggested to address ethical concerns?
What specific examples of AI capabilities were mentioned?
What was the tone of the conversation between the speakers?
"""
    with open(sample_questions_path, "w") as f:
        f.write(sample_questions)

# App title and introduction
st.markdown("<h1 class='main-header'>Gemini Transcript Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered insights from video transcripts using Google's Gemini Pro</p>", unsafe_allow_html=True)

# Create main tabs for workflow
setup_tab, process_tab, chunks_tab, single_tab, batch_tab, stats_tab = st.tabs([
    "🔧 Setup", "📝 Process Transcript", "🧩 Chunk Analysis", "❓ Single Query", "📊 Batch Analysis", "📈 Stats"
])

with setup_tab:
    st.markdown("<h2 class='section-header'>Configuration</h2>", unsafe_allow_html=True)
    
    # API Key configuration
    api_key_col, info_col = st.columns([2, 1])
    with api_key_col:
        api_key = st.text_input("Gemini API Key", type="password", 
                                help="Your Google API key with access to the Gemini API")
        if st.button("Set API Key", use_container_width=True):
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                st.session_state.api_key_set = True
                st.success("✅ API Key set successfully!")
            else:
                st.error("⚠️ Please enter a valid API Key")
    
    with info_col:
        st.markdown("""
        <div class='info-box'>
            <b>Why do I need an API key?</b><br>
            The Gemini API requires authentication via a valid API key from Google AI Studio.
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Configuration
    st.markdown("<h3 class='section-header'>Advanced Configuration</h3>", unsafe_allow_html=True)
    
    with st.expander("Chunking & Context Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", 100, 500, 300, 
                                help="Number of tokens per chunk")
            chunk_overlap = st.slider("Chunk Overlap", 0, 100, 50,
                                    help="Number of tokens that overlap between chunks")
        
        with col2:
            max_context_chunks = st.slider("Max Context Chunks", 3, 15, 8,
                                        help="Maximum number of chunks to include in context")
            semantic_chunking = st.checkbox("Use Semantic Chunking", True,
                                            help="Use semantic boundaries for chunking instead of fixed size")

with process_tab:
    st.markdown("<h2 class='section-header'>Process Your Transcript</h2>", unsafe_allow_html=True)
    
    # Transcript upload interface
    upload_col, info_col = st.columns([2, 1])
    
    with upload_col:
        uploaded_file = st.file_uploader("Upload a transcript file", type=["txt"], 
                                        help="Upload a transcript in plain text format")
        
        sample_checkbox = st.checkbox("Use sample transcript instead", 
                                     help="Use a pre-loaded sample transcript")
    
    with info_col:
        st.markdown("""
        <div class='info-box'>
            <b>Transcript Requirements</b><br>
            • Plain text format (.txt)<br>
            • Ideally with timestamps [HH:MM:SS]<br>
            • One speaker per line<br>
            • Maximum 5 hours of content
        </div>
        """, unsafe_allow_html=True)
    
    # Process button only if API key is set
    if st.session_state.api_key_set:
        if uploaded_file or sample_checkbox:
            process_button = st.button("Process Transcript", type="primary", use_container_width=True)
            
            if process_button:
                with st.container():
                    st.markdown("<h3 class='section-header'>Processing Pipeline</h3>", unsafe_allow_html=True)
                    
                    # Determine transcript path
                    if sample_checkbox:
                        transcript_path = Path("samples/sample_transcript.txt")
                        if not transcript_path.exists():
                            st.error("⚠️ Sample transcript not found. Please upload your own transcript.")
                            st.stop()
                    else:
                        # Save uploaded file
                        transcript_path = Path("uploaded_transcript.txt")
                        with open(transcript_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Get configuration from session state
                    config = {
                        "chunk_size": st.session_state.get("chunk_size", 300),
                        "chunk_overlap": st.session_state.get("chunk_overlap", 50),
                        "max_context_chunks": st.session_state.get("max_context_chunks", 8),
                        "use_semantic_chunking": st.session_state.get("semantic_chunking", True),
                        "api_key": os.environ.get("GEMINI_API_KEY"),
                        "model_name": "gemini-1.5-pro",
                        "context_weights": st.session_state.context_weights
                    }
                    
                    # Process transcript with visual pipeline
                    try:
                        process_transcript_pipeline(transcript_path, config)
                        
                        # Show transcript preview
                        with st.expander("Transcript Preview", expanded=True):
                            # Read the first 1000 characters of the transcript
                            with open(transcript_path, "r") as f:
                                transcript_text = f.read(1000)
                            
                            st.markdown(f"```\n{transcript_text}...\n```")
                            st.markdown(f"**Full transcript length:** {os.path.getsize(transcript_path)} bytes")
                        
                        # Show pipeline summary
                        st.markdown(
                            f"""
                            <div class='card' style="background-color: black;">
                                <h4>Processing Summary</h4>
                                <ul>
                                    <li><b>Chunks created:</b> {len(st.session_state.processor.chunk_manager.chunks)}</li>
                                    <li><b>Chunk size:</b> {config['chunk_size']} tokens</li>
                                    <li><b>Chunk overlap:</b> {config['chunk_overlap']} tokens</li>
                                    <li><b>Using semantic chunking:</b> {config['use_semantic_chunking']}</li>
                                </ul>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Show next steps
                        st.success("✅ Transcript processed successfully! Now you can explore chunks or ask questions.")
                        st.markdown("""
                        <div class='info-box'>
                            <b>Next Steps:</b><br>
                            1. Go to the "Chunk Analysis" tab to inspect how your transcript was segmented<br>
                            2. Try asking a question in the "Single Query" tab<br>
                            3. Or process multiple questions in the "Batch Analysis" tab
                        </div>
                        """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"⚠️ Error processing transcript: {e}")
        else:
            st.info("Please upload a transcript file or select the sample transcript option")
    else:
        st.warning("⚠️ Please set your API Key in the Setup tab first")

with chunks_tab:
    st.markdown("<h2 class='section-header'>Chunk Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_transcript and st.session_state.processor:
        # Add visualize checkbox
        visualize = st.checkbox("Visualize chunks", value=st.session_state.visualize_chunks,
                                help="Show visual representation of chunks")
        st.session_state.visualize_chunks = visualize
        
        # Get chunks from processor
        chunks = st.session_state.processor.chunk_manager.chunks
        
        # Visualization options
        if visualize and st.session_state.chunk_vis_data is not None:
            st.markdown("<h3 class='section-header'>Chunk Visualization</h3>", unsafe_allow_html=True)
            
            # Create visualization
            df = st.session_state.chunk_vis_data
            
            # Create interactive visualizations
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Chunk Length Distribution", "Chunks Timeline"),
                vertical_spacing=0.1,
                specs=[[{"type": "bar"}], [{"type": "scatter"}]]
            )
            
            # Bar chart for chunk lengths
            fig.add_trace(
                go.Bar(
                    x=df['Chunk ID'], 
                    y=df['Length'],
                    marker_color='#1E88E5',
                    name="Chunk Length"
                ),
                row=1, col=1
            )
            
            # Scatter plot for timestamps
            fig.add_trace(
                go.Scatter(
                    x=df['Chunk ID'], 
                    y=df['Timestamp (s)'],
                    mode='markers+lines',
                    marker=dict(size=10, color='#4CAF50'),
                    name="Timestamps"
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=False,
                title_text="Chunk Analysis"
            )
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Entity distribution
            st.markdown("<h3 class='section-header'>Entity Distribution</h3>", unsafe_allow_html=True)
            
            # Collect entities from all chunks
            all_entities = {}
            for chunk in chunks:
                if hasattr(chunk, 'entities') and chunk.entities:
                    for entity in chunk.entities:
                        if entity in all_entities:
                            all_entities[entity] += 1
                        else:
                            all_entities[entity] = 1
            
            # Create bar chart of top entities
            if all_entities:
                # Get top 15 entities
                top_entities = dict(sorted(all_entities.items(), key=lambda x: x[1], reverse=True)[:15])
                
                # Create bar chart
                fig = px.bar(
                    x=list(top_entities.keys()),
                    y=list(top_entities.values()),
                    title="Top Entities Across Chunks",
                    labels={"x": "Entity", "y": "Frequency"},
                    color_discrete_sequence=['#FF9800']
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No entities found in chunks")
        
        # Create chunk explorer
        st.markdown("<h3 class='section-header'>Chunk Explorer</h3>", unsafe_allow_html=True)
        
        # Select chunks to view
        selected_chunk = st.slider(
            "Select chunk ID to explore", 
            min_value=0, 
            max_value=len(chunks)-1 if chunks else 0,
            value=0
        )
        
        if chunks:
            # Get selected chunk
            chunk = chunks[selected_chunk]
            
            # Display chunk details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunk ID", chunk.chunk_id)
            with col2:
                st.metric("Length", len(chunk.text))
            with col3:
                if hasattr(chunk, 'entities'):
                    st.metric("Entities", len(chunk.entities))
                else:
                    st.metric("Entities", 0)
            
            # Display chunk content
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Chunk Content")
            st.markdown(f"```\n{chunk.text}\n```")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display chunk metadata
            with st.expander("Chunk Metadata"):
                metadata = {}
                
                # Add common attributes
                metadata["chunk_id"] = chunk.chunk_id
                metadata["length"] = len(chunk.text)
                
                # Add optional attributes
                if hasattr(chunk, 'start_time'):
                    metadata["start_time"] = chunk.start_time
                if hasattr(chunk, 'end_time'):
                    metadata["end_time"] = chunk.end_time
                if hasattr(chunk, 'entities'):
                    metadata["entities"] = chunk.entities
                if hasattr(chunk, 'topics'):
                    metadata["topics"] = chunk.topics
                
                # Display as JSON
                st.json(metadata)
    else:
        st.info("Please process a transcript in the Process Transcript tab first")

with single_tab:
    st.markdown("<h2 class='section-header'>Ask a Question</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_transcript and st.session_state.processor:
        # Question input area
        query = st.text_area("Your question about the transcript:", 
                             placeholder="Enter your question here...",
                             height=100)
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.markdown("<h4>Context Selection</h4>", unsafe_allow_html=True)
            max_chunks = st.slider("Number of context chunks", 
                                  min_value=2, max_value=15, value=8,
                                  help="Maximum number of chunks to include as context")
            
            st.markdown("<h4>Response Settings</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                confidence_threshold = st.slider("Confidence threshold", 
                                               min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                                               help="Minimum confidence score to accept an answer")
            
            with col2:
                show_confidence = st.checkbox("Show confidence scores", value=True,
                                           help="Display confidence scores with the answer")
        
        # Process question
        if query:
            process_col, debug_col = st.columns([2, 1])
            
            with process_col:
                if st.button("Submit Question", type="primary", use_container_width=True):
                    st.session_state.last_query = query
                    
                    with st.spinner("Analyzing transcript and generating answer..."):
                        # Process single question
                        try:
                            # Get answer from API client
                            result = process_single_query(
                                query=query,
                                processor=st.session_state.processor,
                                api_client=st.session_state.api_client,
                                max_chunks=max_chunks
                            )
                            
                            # Display result
                            st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
                            st.markdown("<h3>Answer</h3>", unsafe_allow_html=True)
                            
                            # Format the answer
                            st.markdown(result["answer"])
                            
                            # Show confidence if enabled
                            if show_confidence and "confidence" in result:
                                confidence = result["confidence"]
                                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                                st.markdown(f"<div class='confidence-score' style='color:{confidence_color}'>Confidence: {confidence:.2f}</div>", 
                                           unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Store for future reference
                            st.session_state.last_result = result
                        
                        except Exception as e:
                            st.error(f"Error processing question: {e}")
            
            with debug_col:
                if st.checkbox("Show debug information", value=False):
                    st.markdown("<h4>Context Selection Debug</h4>", unsafe_allow_html=True)
                    
                    # Show information about chunks if available
                    if hasattr(st.session_state, 'last_result') and 'context_chunks' in st.session_state.last_result:
                        chunks = st.session_state.last_result['context_chunks']
                        st.markdown(f"**Selected {len(chunks)} chunks for context**")
                        
                        for i, chunk_info in enumerate(chunks):
                            with st.expander(f"Chunk {chunk_info['chunk_id']} (Score: {chunk_info['score']:.2f})"):
                                st.text(chunk_info['text'][:150] + "...")
                    
                    # Show query complexity if available
                    if hasattr(st.session_state, 'last_result') and 'query_complexity' in st.session_state.last_result:
                        complexity = st.session_state.last_result['query_complexity']
                        st.metric("Query Complexity", f"{complexity:.2f}")
                        
                        # Show weights used
                        if 'weights_used' in st.session_state.last_result:
                            weights = st.session_state.last_result['weights_used']
                            st.json(weights)
    else:
        st.info("Please process a transcript in the Process Transcript tab first")
        
with batch_tab:
    st.markdown("<h2 class='section-header'>Process Multiple Questions</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_transcript and st.session_state.processor:
        # Input methods
        input_method = st.radio(
            "How would you like to input questions?",
            ["Text Input", "CSV Upload", "Sample Questions"]
        )
        
        questions = []
        
        if input_method == "Text Input":
            questions_text = st.text_area(
                "Enter your questions (one per line):",
                height=200,
                placeholder="What is the main topic discussed?\nWho are the speakers?\nWhat conclusions were drawn?"
            )
            
            if questions_text:
                questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
                
        elif input_method == "CSV Upload":
            uploaded_csv = st.file_uploader("Upload CSV with questions", type=["csv"])
            
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    if "question" in df.columns:
                        questions = df["question"].tolist()
                    else:
                        # Assume first column has questions
                        questions = df.iloc[:, 0].tolist()
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        elif input_method == "Sample Questions":
            # Load sample questions if they exist
            sample_path = Path("samples/sample_questions.csv")
            if sample_path.exists():
                try:
                    df = pd.read_csv(sample_path)
                    questions = df["question"].tolist()
                    
                    st.markdown(
                        "<div class='info-box'>Loaded sample questions from samples/sample_questions.csv</div>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error loading sample questions: {e}")
            else:
                st.info("Sample questions file not found. Using default questions instead.")
                questions = [
                    "What is the main topic of this transcript?",
                    "Who are the main speakers in this discussion?",
                    "What are the key points made in this transcript?",
                    "What conclusions or decisions were reached?",
                    "What is the overall tone of the conversation?"
                ]
        
        # Show questions preview
        if questions:
            st.markdown(f"**{len(questions)} questions to process:**")
            
            preview_questions = questions[:5]
            for i, q in enumerate(preview_questions, 1):
                st.markdown(f"{i}. {q}")
            
            if len(questions) > 5:
                st.markdown(f"*... and {len(questions) - 5} more questions*")
        
            # Batch processing options
            with st.expander("Batch Processing Options"):
                st.markdown("<h4>Processing Settings</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    max_chunks = st.slider("Max context chunks", 
                                         min_value=2, max_value=15, value=8,
                                         help="Maximum number of chunks to include as context")
                
                with col2:
                    batch_size = st.slider("Batch size", 
                                         min_value=1, max_value=10, value=5,
                                         help="Number of questions to process in parallel")
                
                enable_dependencies = st.checkbox("Use dependency tracking", value=True,
                                               help="Track dependencies between questions")
                
                st.markdown("<h4>Output Options</h4>", unsafe_allow_html=True)
                export_format = st.selectbox(
                    "Export format",
                    ["JSON", "CSV", "Text"],
                    help="Format to export results"
                )
            
            # Process batch
            if st.button("Process Batch", type="primary", use_container_width=True):
                # Start batch processing
                with st.spinner(f"Processing {len(questions)} questions..."):
                    try:
                        # Create progress bar
                        progress = st.progress(0)
                        status_text = st.empty()
                        
                        # Create result container
                        results_container = st.container()
                        
                        # Process questions
                        results = process_question_batch(
                            questions=questions,
                            processor=st.session_state.processor,
                            api_client=st.session_state.api_client,
                            batch_size=batch_size,
                            max_chunks=max_chunks,
                            use_dependencies=enable_dependencies,
                            progress_callback=lambda i, total: (
                                progress.progress(i/total),
                                status_text.text(f"Processed {i}/{total} questions...")
                            )
                        )
                        
                        # Complete progress
                        progress.progress(1.0)
                        status_text.text("Processing complete!")
                        
                        # Store results
                        st.session_state.batch_results = results
                        
                        # Show results table
                        with results_container:
                            st.markdown("<h3 class='section-header'>Results</h3>", unsafe_allow_html=True)
                            
                            # Create results table
                            results_df = pd.DataFrame([
                                {
                                    "Question": result["question"],
                                    "Answer": result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"],
                                    "Confidence": result.get("confidence", "-")
                                }
                                for result in results
                            ])
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Export options
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                if export_format == "JSON":
                                    json_str = json.dumps(results, indent=2)
                                    st.download_button(
                                        "Download JSON Results",
                                        data=json_str,
                                        file_name="transcript_qa_results.json",
                                        mime="application/json"
                                    )
                                elif export_format == "CSV":
                                    csv_buffer = StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    st.download_button(
                                        "Download CSV Results",
                                        data=csv_buffer.getvalue(),
                                        file_name="transcript_qa_results.csv",
                                        mime="text/csv"
                                    )
                                else:  # Text format
                                    text_output = "\n\n".join([
                                        f"Q: {result['question']}\nA: {result['answer']}"
                                        for result in results
                                    ])
                                    st.download_button(
                                        "Download Text Results",
                                        data=text_output,
                                        file_name="transcript_qa_results.txt",
                                        mime="text/plain"
                                    )
                            
                            with export_col2:
                                # Session saving option
                                st.download_button(
                                    "Save Session for Later",
                                    data=json.dumps({
                                        "timestamp": datetime.now().isoformat(),
                                        "questions": questions,
                                        "results": results,
                                        "config": {
                                            "batch_size": batch_size,
                                            "max_chunks": max_chunks,
                                            "use_dependencies": enable_dependencies
                                        }
                                    }, indent=2),
                                    file_name="qa_session.json",
                                    mime="application/json"
                                )
                            
                            # Visualization of results
                            st.markdown("<h3 class='section-header'>Analysis</h3>", unsafe_allow_html=True)
                            
                            # Get confidence scores
                            confidence_scores = [result.get("confidence", 0) for result in results]
                            
                            # Create confidence histogram
                            if any(confidence_scores):
                                fig = px.histogram(
                                    confidence_scores,
                                    nbins=10,
                                    labels={"value": "Confidence Score", "count": "Number of Questions"},
                                    title="Confidence Score Distribution",
                                    color_discrete_sequence=['#4CAF50']
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show dependencies if they exist
                            if enable_dependencies and any("dependencies" in result for result in results):
                                st.markdown("<h4>Question Dependencies</h4>", unsafe_allow_html=True)
                                
                                # Collect dependency data
                                dependencies = []
                                for i, result in enumerate(results):
                                    if "dependencies" in result and result["dependencies"]:
                                        for dep in result["dependencies"]:
                                            dependencies.append({
                                                "from": dep["question_idx"],
                                                "to": i,
                                                "strength": dep.get("strength", 1)
                                            })
                                
                                if dependencies:
                                    # Create dependency network graph
                                    # (simplified version since plotly doesn't have direct network graph)
                                    dep_fig = go.Figure()
                                    
                                    # Add edges
                                    for dep in dependencies:
                                        dep_fig.add_trace(
                                            go.Scatter(
                                                x=[dep["from"], dep["to"]],
                                                y=[0, 0],
                                                mode="lines",
                                                line=dict(width=dep["strength"] * 3, color="#2196F3"),
                                                showlegend=False
                                            )
                                        )
                                    
                                    # Add nodes
                                    for i in range(min(10, len(questions))):
                                        dep_fig.add_trace(
                                            go.Scatter(
                                                x=[i],
                                                y=[0],
                                                mode="markers+text",
                                                marker=dict(size=15, color="#FF5722"),
                                                text=[f"Q{i+1}"],
                                                textposition="top center",
                                                showlegend=False
                                            )
                                        )
                                    
                                    # Update layout
                                    dep_fig.update_layout(
                                        title="Question Dependencies (First 10 Questions)",
                                        xaxis=dict(showticklabels=False, showgrid=False),
                                        yaxis=dict(showticklabels=False, showgrid=False),
                                        height=200,
                                        margin=dict(l=20, r=20, t=40, b=20)
                                    )
                                    
                                    st.plotly_chart(dep_fig, use_container_width=True)
                                else:
                                    st.info("No dependencies detected between questions")
                    
                    except Exception as e:
                        st.error(f"Error processing batch: {e}")
                        st.exception(e)
    else:
        st.info("Please process a transcript in the Process Transcript tab first")

with stats_tab:
    st.markdown("<h2 class='section-header'>System Performance Analytics</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_transcript:
        # Create a dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "API Calls",
                st.session_state.get("api_calls", 0),
                help="Total number of API calls made to Gemini"
            )
        
        with col2:
            cache_hit_rate = 0
            if (st.session_state.get("cache_hits", 0) + st.session_state.get("cache_misses", 0)) > 0:
                cache_hit_rate = st.session_state.get("cache_hits", 0) / (
                    st.session_state.get("cache_hits", 0) + st.session_state.get("cache_misses", 0)
                ) * 100
            
            st.metric(
                "Cache Hit Rate",
                f"{cache_hit_rate:.1f}%",
                help="Percentage of queries served from cache"
            )
        
        with col3:
            st.metric(
                "Tokens Saved",
                st.session_state.get("total_saved_tokens", 0),
                help="Estimated number of tokens saved by caching"
            )
        
        with col4:
            if "processor" in st.session_state:
                st.metric(
                    "Total Chunks",
                    len(st.session_state.processor.chunk_manager.chunks),
                    help="Number of chunks created from transcript"
                )
        
        # Create tabs for detailed analytics
        perf_tab, cache_tab, weights_tab = st.tabs(["Performance", "Cache Analysis", "Context Weights"])
        
        with perf_tab:
            st.markdown("<h3 class='section-header'>Performance Analysis</h3>", unsafe_allow_html=True)
            
            # Create sample performance data if we don't have real data yet
            if "performance_data" not in st.session_state:
                # Create sample data for demonstration
                st.session_state.performance_data = {
                    "query_times": [],
                    "context_selection_times": [],
                    "api_response_times": []
                }
            
            perf_data = st.session_state.performance_data
            
            if perf_data["query_times"]:
                # Create performance visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Query Processing Time", "Time Breakdown"),
                    specs=[[{"type": "scatter"}, {"type": "bar"}]]
                )
                
                # Query time trend
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(perf_data["query_times"]))),
                        y=perf_data["query_times"],
                        mode="lines+markers",
                        name="Query Time",
                        line=dict(color="#1E88E5")
                    ),
                    row=1, col=1
                )
                
                # Time breakdown
                avg_context_time = sum(perf_data["context_selection_times"]) / len(perf_data["context_selection_times"]) if perf_data["context_selection_times"] else 0
                avg_api_time = sum(perf_data["api_response_times"]) / len(perf_data["api_response_times"]) if perf_data["api_response_times"] else 0
                
                fig.add_trace(
                    go.Bar(
                        x=["Context Selection", "API Response"],
                        y=[avg_context_time, avg_api_time],
                        marker_color=["#4CAF50", "#FF9800"]
                    ),
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display averages
                st.markdown(
                    f"""
                    <div class='card'>
                        <h4>Average Response Times</h4>
                        <ul>
                            <li><b>Overall query time:</b> {sum(perf_data["query_times"]) / len(perf_data["query_times"]):.2f}s</li>
                            <li><b>Context selection:</b> {avg_context_time:.2f}s</li>
                            <li><b>API response:</b> {avg_api_time:.2f}s</li>
                        </ul>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.info("Process some queries to see performance data")
        
        with cache_tab:
            st.markdown("<h3 class='section-header'>Cache Analysis</h3>", unsafe_allow_html=True)
            
            # Create cache visualization
            fig = go.Figure()
            
            # Cache hits vs misses pie chart
            fig.add_trace(
                go.Pie(
                    labels=["Cache Hits", "Cache Misses"],
                    values=[st.session_state.get("cache_hits", 0), st.session_state.get("cache_misses", 0)],
                    marker=dict(colors=["#4CAF50", "#F44336"])
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Cache Hits vs Misses",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cache efficiency stats
            total_queries = st.session_state.get("cache_hits", 0) + st.session_state.get("cache_misses", 0)
            
            if total_queries > 0:
                savings_percentage = st.session_state.get("cache_hits", 0) / total_queries * 100
                
                st.markdown(
                    f"""
                    <div class='card'>
                        <h4>Cache Efficiency</h4>
                        <ul>
                            <li><b>Total queries:</b> {total_queries}</li>
                            <li><b>Cache hits:</b> {st.session_state.get("cache_hits", 0)}</li>
                            <li><b>Cache misses:</b> {st.session_state.get("cache_misses", 0)}</li>
                            <li><b>Cache hit rate:</b> {savings_percentage:.1f}%</li>
                            <li><b>Estimated token savings:</b> {st.session_state.get("total_saved_tokens", 0)}</li>
                        </ul>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.info("No cache data available yet")
        
        with weights_tab:
            st.markdown("<h3 class='section-header'>Context Selection Weights</h3>", unsafe_allow_html=True)
            
            # Allow user to tune context weights
            st.markdown("""
            Adjust the weights to fine-tune how context is selected for queries.
            These weights determine which factors are most important when ranking chunks.
            """)
            
            # Show current weights
            weights = st.session_state.context_weights
            
            # Allow weight adjustment
            col1, col2 = st.columns(2)
            
            with col1:
                new_weights = {}
                
                new_weights["recency"] = st.slider(
                    "Recency", 
                    min_value=0.0, max_value=1.0, value=weights["recency"], step=0.05,
                    help="How much to prioritize chunks that are closer to the query in time"
                )
                
                new_weights["relevance"] = st.slider(
                    "Relevance", 
                    min_value=0.0, max_value=1.0, value=weights["relevance"], step=0.05,
                    help="How much to prioritize chunks that are semantically relevant to the query"
                )
                
                new_weights["entity"] = st.slider(
                    "Entity Matching", 
                    min_value=0.0, max_value=1.0, value=weights["entity"], step=0.05,
                    help="How much to prioritize chunks containing entities mentioned in the query"
                )
            
            with col2:
                new_weights["topic"] = st.slider(
                    "Topic Relevance", 
                    min_value=0.0, max_value=1.0, value=weights.get("topic", 0), step=0.05,
                    help="How much to prioritize chunks with topics relevant to the query"
                )
                
                new_weights["concept"] = st.slider(
                    "Concept Coverage", 
                    min_value=0.0, max_value=1.0, value=weights.get("concept", 0), step=0.05,
                    help="How much to prioritize chunks that cover broader concepts for complex queries"
                )
            
            # Apply new weights button
            if st.button("Apply Weight Changes", use_container_width=True):
                st.session_state.context_weights = new_weights
                
                # Update context manager if it exists
                if "context_manager" in st.session_state:
                    st.session_state.context_manager.weights = new_weights
                
                st.success("Context selection weights updated successfully!")
            
            # Show weight distribution visualization
            fig = px.bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                labels={"x": "Factor", "y": "Weight"},
                title="Context Selection Weight Distribution",
                color=list(weights.values()),
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(
                xaxis_title="Factor",
                yaxis_title="Weight",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please process a transcript in the Process Transcript tab first")
