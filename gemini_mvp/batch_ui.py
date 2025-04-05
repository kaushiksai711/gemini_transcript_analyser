# gemini_mvp/batch_ui.py - NOT ACTIVELY USED
# Provides UI components for batch processing (not used in current implementation)

"""
Batch UI module for visual interface components.

This module implements UI elements for batch processing visualization
and interactive result exploration.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .batch_processor import BatchProcessor
from .long_context_handler import LongContextHandler
from .context_cache import ContextCache
from .api_client import GeminiClient
from .context_manager import ContextManager
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)

def render_batch_ui(
    processor,
    api_client: GeminiClient,
    context_manager: ContextManager,
    cache_manager: CacheManager
):
    """
    Render the enhanced batch processing UI.
    
    Args:
        processor: Transcript processor instance
        api_client: API client instance
        context_manager: Context manager instance
        cache_manager: Cache manager instance
    """
    st.markdown("<h2 class='section-header'>Enhanced Batch Processing</h2>", unsafe_allow_html=True)
    
    if not hasattr(st.session_state, "batch_processor"):
        # Initialize specialized components
        st.session_state.batch_processor = BatchProcessor(
            api_client=api_client,
            query_processor=processor.query_processor,
            context_manager=context_manager,
            cache_manager=cache_manager
        )
        
        st.session_state.long_context_handler = LongContextHandler(
            chunk_manager=processor.chunk_manager,
            cache_manager=cache_manager
        )
        
        st.session_state.context_cache = ContextCache(
            cache_manager=cache_manager
        )
    
    # Create tabs for different batch operations
    batch_tabs = st.tabs([
        "1️⃣ Batch Questions", 
        "2️⃣ Dependency Analysis", 
        "3️⃣ Performance Metrics",
        "4️⃣ Cache Analysis"
    ])
    
    # Batch Questions Tab
    with batch_tabs[0]:
        render_batch_questions_tab(processor)
    
    # Dependency Analysis Tab
    with batch_tabs[1]:
        render_dependency_analysis_tab()
    
    # Performance Metrics Tab
    with batch_tabs[2]:
        render_performance_metrics_tab()
    
    # Cache Analysis Tab
    with batch_tabs[3]:
        render_cache_analysis_tab()


def render_batch_questions_tab(processor):
    """Render the batch questions input and processing interface."""
    st.markdown("<h3>Process Multiple Questions Efficiently</h3>", unsafe_allow_html=True)
    
    # Input methods
    input_method = st.radio(
        "How would you like to input questions?",
        ["Text Input", "CSV Upload", "JSON Upload", "Sample Questions"]
    )
    
    questions = []
    
    if input_method == "Text Input":
        questions_text = st.text_area(
            "Enter questions (one per line):",
            height=150,
            placeholder="What is the main topic discussed?\nWho are the speakers?\nWhat conclusions were reached?"
        )
        
        if questions_text:
            questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
    
    elif input_method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV file with questions", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if "question" in df.columns:
                    questions = df["question"].dropna().tolist()
                else:
                    # Assume first column contains questions
                    questions = df.iloc[:, 0].dropna().tolist()
                    
                st.success(f"Successfully loaded {len(questions)} questions from CSV")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    
    elif input_method == "JSON Upload":
        uploaded_file = st.file_uploader("Upload JSON file with questions", type="json")
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    if all(isinstance(item, str) for item in data):
                        questions = data
                    elif all(isinstance(item, dict) for item in data):
                        # Try to extract questions from objects
                        if all("question" in item for item in data):
                            questions = [item["question"] for item in data]
                        else:
                            st.warning("JSON format not recognized. Each item should have a 'question' field.")
                elif isinstance(data, dict) and "questions" in data:
                    questions = data["questions"]
                else:
                    st.warning("JSON format not recognized")
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
    
    elif input_method == "Sample Questions":
        sample_set = st.selectbox(
            "Select a sample question set:",
            ["Video Topic Questions", "Speaker Analysis", "Concept Exploration", "Timeline Questions"]
        )
        
        if sample_set == "Video Topic Questions":
            questions = [
                "What is the main topic of this video?",
                "What are the key points discussed?",
                "How does the speaker introduce the topic?",
                "What examples are provided to illustrate the main concepts?",
                "What conclusions does the speaker reach at the end?",
                "How does this topic relate to current trends?"
            ]
        elif sample_set == "Speaker Analysis":
            questions = [
                "Who are the speakers in this video?",
                "What is Speaker 1's main argument?",
                "Does Speaker 2 agree with Speaker 1's points?",
                "What expertise or background does each speaker have?",
                "Are there any disagreements between speakers?",
                "What credentials or evidence do speakers cite to support their claims?"
            ]
        elif sample_set == "Concept Exploration":
            questions = [
                "Define the concept of artificial intelligence as explained in the video",
                "How is machine learning different from traditional programming?",
                "What ethical concerns related to AI are mentioned?",
                "How might these technologies develop in the next decade according to the speakers?",
                "What limitations of current AI systems are discussed?",
                "How do the speakers suggest addressing potential risks?"
            ]
        elif sample_set == "Timeline Questions":
            questions = [
                "What happened in the first 5 minutes of the video?",
                "When does the speaker introduce the main problem?",
                "What topics are covered in the middle section (around 15-20 minutes)?",
                "When are case studies or examples presented?",
                "What information is shared in the conclusion?",
                "At what point is the most important insight revealed?"
            ]
    
    # Display the list of questions
    if questions:
        st.markdown(f"#### Processing {len(questions)} Questions")
        
        with st.expander("View all questions", expanded=len(questions) < 10):
            for i, q in enumerate(questions, 1):
                st.markdown(f"{i}. {q}")
        
        # Batch processing options
        st.markdown("### Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_dependencies = st.checkbox("Track question dependencies", value=True,
                                        help="Analyze dependencies between questions for optimal ordering")
            enable_cache = st.checkbox("Enable context caching", value=True,
                                    help="Cache context chunks to reduce token usage")
        
        with col2:
            batch_size = st.slider("Batch size", 1, 20, 5,
                                help="Number of questions to process in each batch")
            optimize_context = st.checkbox("Optimize context for batches", value=True,
                                        help="Find optimal context chunks for related questions")
        
        # Execution button
        process_button = st.button("Process Batch", type="primary", use_container_width=True)
        
        if process_button:
            # Prepare the results container
            results_container = st.container()
            
            # Ensure processor is initialized
            if not hasattr(st.session_state, "batch_processor"):
                st.error("Batch processor not initialized properly")
                return
            
            # Setup progress tracking
            progress = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message=None):
                """Update progress bar and status message."""
                progress.progress(current / total)
                if message:
                    status_text.text(message)
            
            try:
                # Process the batch
                with st.spinner("Processing questions..."):
                    status_text.text("Analyzing question dependencies...")
                    
                    # Get available chunks
                    available_chunks = processor.chunk_manager.chunks
                    
                    if optimize_context and hasattr(st.session_state, "long_context_handler"):
                        # Process transcript for long context handling if not already done
                        if not st.session_state.long_context_handler.context_windows:
                            status_text.text("Optimizing transcript for long context handling...")
                            transcript_text = processor.chunk_manager.transcript_text
                            st.session_state.long_context_handler.process_transcript(transcript_text)
                        
                        # Optimize context for the batch
                        status_text.text("Finding optimal context for question batch...")
                        optimized_chunks = st.session_state.long_context_handler.optimize_context_for_batch(
                            questions=questions
                        )
                        available_chunks = optimized_chunks
                    
                    # Generate a batch ID
                    batch_id = f"batch_{int(time.time())}"
                    
                    # Process the batch
                    status_text.text(f"Processing {len(questions)} questions...")
                    
                    batch_results, metrics = st.session_state.batch_processor.process_batch(
                        questions=questions,
                        available_chunks=available_chunks,
                        batch_id=batch_id
                    )
                    
                    # Save the batch results and metrics in session state
                    st.session_state.last_batch_id = batch_id
                    st.session_state.last_batch_results = batch_results
                    st.session_state.last_batch_metrics = metrics
                    
                    # Create dependency visualization
                    viz_path = st.session_state.batch_processor.visualize_dependency_graph()
                    st.session_state.last_dependency_graph = viz_path
                    
                    # Complete the progress
                    progress.progress(1.0)
                    status_text.text(f"Completed processing {len(questions)} questions!")
                    
                    # Display results
                    with results_container:
                        st.markdown("## Batch Results")
                        
                        # Create results table
                        results_df = pd.DataFrame([
                            {
                                "Question": r.get("question", f"Question {i+1}"),
                                "Answer": r.get("answer", "No answer generated"),
                                "Confidence": f"{r.get('confidence', 0)*100:.1f}%",
                                "Response Time": f"{r.get('response_time', 0):.2f}s",
                                "Success": "✅" if r.get("success", False) else "❌",
                            }
                            for i, r in enumerate(batch_results)
                        ])
                        
                        # Display the results table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Display summary metrics
                        st.markdown("### Performance Summary")
                        
                        metrics_cols = st.columns(4)
                        
                        with metrics_cols[0]:
                            st.metric("Total Questions", metrics.total_questions)
                            st.metric("Successful", metrics.successful_questions)
                        
                        with metrics_cols[1]:
                            st.metric("Failed", metrics.failed_questions)
                            st.metric("Batches Created", metrics.batches_created)
                        
                        with metrics_cols[2]:
                            st.metric("Cache Hits", metrics.cache_hits)
                            token_savings = f"{(metrics.tokens_saved / max(1, metrics.tokens_saved + metrics.tokens_used)) * 100:.1f}%"
                            st.metric("Token Savings", token_savings)
                        
                        with metrics_cols[3]:
                            st.metric("API Calls", metrics.api_calls)
                            st.metric("Avg Response Time", f"{metrics.avg_response_time:.2f}s")
                        
                        # Option to export results
                        export_col1, export_col2 = st.columns(2)
                        
                        with export_col1:
                            if st.button("Export Results to JSON", type="secondary"):
                                results_path = st.session_state.batch_processor.save_batch_results(
                                    batch_results, batch_id
                                )
                                st.success(f"Results exported to {results_path}")
                                # Allow downloading the file
                                with open(results_path, "r") as f:
                                    st.download_button(
                                        "Download JSON",
                                        f,
                                        file_name=f"batch_results_{batch_id}.json",
                                        mime="application/json"
                                    )
                        
                        with export_col2:
                            if st.button("Export as CSV", type="secondary"):
                                # Create a more detailed dataframe for export
                                export_df = pd.DataFrame([
                                    {
                                        "question": r.get("question", ""),
                                        "answer": r.get("answer", ""),
                                        "confidence": r.get("confidence", 0),
                                        "response_time": r.get("response_time", 0),
                                        "success": r.get("success", False),
                                        "context_chunks": ",".join(map(str, r.get("context_chunks", []))),
                                        "timestamp": r.get("timestamp", "")
                                    }
                                    for r in batch_results
                                ])
                                
                                # Convert to CSV
                                csv = export_df.to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    csv,
                                    file_name=f"batch_results_{batch_id}.csv",
                                    mime="text/csv"
                                )
            
            except Exception as e:
                st.error(f"Error processing batch: {e}")
                st.exception(e)


def render_dependency_analysis_tab():
    """Render dependency analysis visualizations and insights."""
    st.markdown("<h3>Question Dependency Analysis</h3>", unsafe_allow_html=True)
    
    if hasattr(st.session_state, "last_batch_id") and st.session_state.last_batch_id:
        # Display dependency graph if available
        if hasattr(st.session_state, "last_dependency_graph") and st.session_state.last_dependency_graph:
            graph_path = st.session_state.last_dependency_graph
            
            if os.path.exists(graph_path):
                st.markdown("### Question Dependency Graph")
                st.image(graph_path)
                st.caption("This graph shows dependencies between questions. Arrows indicate which questions depend on other questions.")
            
        # Display dependency metrics if available
        if hasattr(st.session_state, "last_batch_metrics") and st.session_state.last_batch_metrics:
            metrics = st.session_state.last_batch_metrics
            
            st.markdown("### Dependency Metrics")
            
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Independent Questions", metrics.independent_questions)
                st.markdown("*Questions that don't depend on other questions*")
            
            with cols[1]:
                st.metric("Dependent Questions", metrics.dependent_questions)
                st.markdown("*Questions that depend on previous questions*")
            
            with cols[2]:
                dep_ratio = f"{(metrics.dependent_questions / max(1, metrics.total_questions)) * 100:.1f}%"
                st.metric("Dependency Ratio", dep_ratio)
                st.markdown("*Percentage of questions with dependencies*")
            
            # Display batch organization
            if hasattr(st.session_state, "batch_processor") and st.session_state.batch_processor:
                st.markdown("### Batch Organization")
                
                # Create a graph showing how questions were batched
                if hasattr(st.session_state.batch_processor, "dependency_graph"):
                    graph = st.session_state.batch_processor.dependency_graph
                    
                    # Get connected components
                    import networkx as nx
                    components = list(nx.weakly_connected_components(graph))
                    
                    # Display information about batches
                    st.markdown(f"Questions were organized into **{metrics.batches_created}** batches based on dependencies.")
                    
                    if len(components) > 1:
                        st.markdown(f"Found **{len(components)}** independent groups of related questions.")
                        
                        # Show the groups
                        for i, component in enumerate(components, 1):
                            with st.expander(f"Group {i}: {len(component)} questions"):
                                for node in sorted(component):
                                    question = graph.nodes[node].get("question", f"Question {node}")
                                    st.markdown(f"- {question}")
                    else:
                        st.markdown("All questions have potential dependencies and were processed as a single group.")
    else:
        st.info("Process a batch of questions first to see dependency analysis.")


def render_performance_metrics_tab():
    """Render performance metrics for batch processing."""
    st.markdown("<h3>Batch Processing Performance</h3>", unsafe_allow_html=True)
    
    if hasattr(st.session_state, "last_batch_metrics") and st.session_state.last_batch_metrics:
        metrics = st.session_state.last_batch_metrics
        
        # Create dashboard layout
        st.markdown("### Key Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create processing time visualization
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.avg_response_time,
                title={"text": "Avg Response Time (seconds)"},
                gauge={
                    "axis": {"range": [0, max(10, metrics.avg_response_time * 1.5)]},
                    "bar": {"color": "royalblue"},
                    "steps": [
                        {"range": [0, 1], "color": "green"},
                        {"range": [1, 3], "color": "yellow"},
                        {"range": [3, 10], "color": "orange"},
                        {"range": [10, max(10, metrics.avg_response_time * 1.5)], "color": "red"}
                    ]
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create success rate visualization
            success_rate = metrics.successful_questions / max(1, metrics.total_questions) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=success_rate,
                title={"text": "Success Rate (%)"},
                delta={"reference": 95, "increasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 50], "color": "red"},
                        {"range": [50, 80], "color": "orange"},
                        {"range": [80, 90], "color": "yellow"},
                        {"range": [90, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 95
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Token usage statistics
        st.markdown("### Token Usage Analytics")
        
        token_cols = st.columns(2)
        
        with token_cols[0]:
            # Token distribution pie chart
            token_data = {
                "Category": ["Tokens Saved", "Tokens Used"],
                "Count": [metrics.tokens_saved, metrics.tokens_used]
            }
            
            token_df = pd.DataFrame(token_data)
            
            fig = px.pie(
                token_df, 
                values="Count", 
                names="Category",
                color="Category",
                color_discrete_map={
                    "Tokens Saved": "green",
                    "Tokens Used": "red"
                },
                title="Token Usage Distribution"
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw numbers
            st.metric("Total Tokens Saved", format(metrics.tokens_saved, ","))
            st.metric("Total Tokens Used", format(metrics.tokens_used, ","))
        
        with token_cols[1]:
            # API call savings
            api_data = {
                "Category": ["API Calls Made", "API Calls Saved"],
                "Count": [metrics.api_calls, metrics.total_questions - metrics.api_calls]
            }
            
            api_df = pd.DataFrame(api_data)
            
            fig = px.bar(
                api_df,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map={
                    "API Calls Made": "darkblue",
                    "API Calls Saved": "green"
                },
                title="API Call Efficiency"
            )
            
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Savings percentage
            api_savings_pct = ((metrics.total_questions - metrics.api_calls) / 
                             max(1, metrics.total_questions)) * 100
            
            st.metric(
                "API Call Savings", 
                f"{api_savings_pct:.1f}%",
                delta=f"{metrics.total_questions - metrics.api_calls} calls"
            )
            
            token_savings_pct = (metrics.tokens_saved / 
                               max(1, metrics.tokens_saved + metrics.tokens_used)) * 100
            
            st.metric(
                "Token Savings", 
                f"{token_savings_pct:.1f}%",
                delta=f"{format(metrics.tokens_saved, ',')} tokens"
            )
    else:
        st.info("Process a batch of questions first to see performance metrics.")


def render_cache_analysis_tab():
    """Render cache analysis and statistics."""
    st.markdown("<h3>Context Cache Analytics</h3>", unsafe_allow_html=True)
    
    if hasattr(st.session_state, "context_cache"):
        cache = st.session_state.context_cache
        
        # Get cache stats
        cache_stats = cache.get_stats()
        
        st.markdown("### Cache Performance")
        
        # Display key cache metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            hit_rate = cache_stats.get("hit_rate", 0)
            st.metric("Cache Hit Rate", f"{hit_rate}%")
            
            hits = cache_stats.get("hits", 0)
            misses = cache_stats.get("misses", 0)
            st.metric("Cache Hits", hits, delta=f"{hits-misses} vs misses")
        
        with metric_cols[1]:
            total_entries = cache_stats.get("total_cached_entries", 0)
            st.metric("Cached Entries", total_entries)
            
            partial_hits = cache_stats.get("partial_hits", 0)
            st.metric("Partial Cache Hits", partial_hits)
        
        with metric_cols[2]:
            tokens_saved = cache_stats.get("tokens_saved", 0)
            st.metric("Tokens Saved", format(tokens_saved, ","))
            
            token_savings_pct = cache_stats.get("token_savings_percent", 0)
            st.metric("Token Savings %", f"{token_savings_pct}%")
        
        with metric_cols[3]:
            api_calls_saved = cache_stats.get("api_calls_saved", 0)
            st.metric("API Calls Saved", api_calls_saved)
            
            cache_size = cache_stats.get("cache_size_bytes", 0) / (1024 * 1024)  # Convert to MB
            st.metric("Cache Size", f"{cache_size:.2f} MB")
        
        # Cache visualization
        st.markdown("### Cache Visualization")
        
        viz_tabs = st.tabs(["Hit Rate", "Query Types", "Token Savings"])
        
        with viz_tabs[0]:
            # Create hit rate visualization
            hit_data = {
                "Category": ["Hits", "Partial Hits", "Misses"],
                "Count": [
                    cache_stats.get("hits", 0),
                    cache_stats.get("partial_hits", 0),
                    cache_stats.get("misses", 0)
                ]
            }
            
            hit_df = pd.DataFrame(hit_data)
            
            fig = px.pie(
                hit_df,
                values="Count",
                names="Category",
                color="Category",
                color_discrete_map={
                    "Hits": "green",
                    "Partial Hits": "orange",
                    "Misses": "red"
                },
                title="Cache Hit Distribution"
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            # Create query type distribution
            query_types = cache_stats.get("query_type_distribution", {})
            
            if query_types:
                query_data = {
                    "Query Type": list(query_types.keys()),
                    "Count": list(query_types.values())
                }
                
                query_df = pd.DataFrame(query_data)
                
                fig = px.bar(
                    query_df,
                    x="Query Type",
                    y="Count",
                    color="Query Type",
                    title="Cache Entries by Query Type"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query type distribution data available")
        
        with viz_tabs[2]:
            # Create token savings visualization
            token_data = {
                "Category": ["Tokens Saved", "Tokens Used"],
                "Count": [
                    cache_stats.get("tokens_saved", 0),
                    cache_stats.get("tokens_used", 0)
                ]
            }
            
            token_df = pd.DataFrame(token_data)
            
            fig = px.bar(
                token_df,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map={
                    "Tokens Saved": "green",
                    "Tokens Used": "red"
                },
                title="Token Usage Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cache management options
        st.markdown("### Cache Management")
        
        manage_cols = st.columns(2)
        
        with manage_cols[0]:
            if st.button("Reset Cache Statistics", type="secondary"):
                cache.reset_stats()
                st.success("Cache statistics reset successfully")
                st.experimental_rerun()
        
        with manage_cols[1]:
            if st.button("Clear Cache", type="secondary"):
                entries_cleared = cache.invalidate_cache()
                st.success(f"Cache cleared successfully. {entries_cleared} entries removed.")
                st.experimental_rerun()
        
        # Export cache info
        if st.button("Export Cache Analytics", type="secondary"):
            export_path = cache.export_cache_info()
            st.success(f"Cache analytics exported to {export_path}")
            
            # Allow downloading the file
            with open(export_path, "r") as f:
                st.download_button(
                    "Download Cache Analytics",
                    f,
                    file_name=f"cache_analytics_{int(time.time())}.json",
                    mime="application/json"
                )
    else:
        st.info("Initialize the context cache first by processing a batch of questions.")
