# gemini_mvp/batch_processor.py - ACTIVELY USED
# Core component for processing batches of questions against context chunks

"""
Batch Processor module for handling batches of questions.

This module implements batch processing logic, context optimization,
and dependency tracking for efficient question handling.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

from .query_processor import QueryProcessor
from .context_manager import ContextManager
from .cache_manager import CacheManager
from .dependency_tracker import DependencyTracker
from .api_client import GeminiClient
from .chunking import Chunk

logger = logging.getLogger(__name__)

@dataclass
class BatchMetrics:
    """Metrics for batch processing performance tracking."""
    
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    cache_hits: int = 0
    tokens_saved: int = 0
    tokens_used: int = 0
    total_time: float = 0
    avg_response_time: float = 0
    batches_created: int = 0
    independent_questions: int = 0
    dependent_questions: int = 0
    context_reuse_count: int = 0
    api_calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_questions": self.total_questions,
            "successful_questions": self.successful_questions,
            "failed_questions": self.failed_questions,
            "cache_hits": self.cache_hits,
            "tokens_saved": self.tokens_saved,
            "tokens_used": self.tokens_used,
            "token_savings_percent": round((self.tokens_saved / max(1, self.tokens_saved + self.tokens_used)) * 100, 2),
            "total_time_seconds": round(self.total_time, 2),
            "avg_response_time_seconds": round(self.avg_response_time, 2),
            "batches_created": self.batches_created,
            "independent_questions": self.independent_questions,
            "dependent_questions": self.dependent_questions,
            "context_reuse_count": self.context_reuse_count,
            "api_calls": self.api_calls,
            "api_calls_saved": self.total_questions - self.api_calls,
            "timestamp": datetime.now().isoformat()
        }


class BatchProcessor:
    """
    Handles efficient batch processing of questions.
    
    Features:
    - Intelligent dependency-based batching
    - Context caching across related questions
    - Token usage optimization
    - Parallel processing of independent questions
    - Visualization of question dependencies
    """
    
    def __init__(
        self,
        api_client: GeminiClient,
        query_processor: QueryProcessor,
        context_manager: ContextManager,
        cache_manager: CacheManager,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            api_client: Gemini API client
            query_processor: Query processor instance
            context_manager: Context manager instance
            cache_manager: Cache manager instance
            config: Configuration options
        """
        self.api_client = api_client
        self.query_processor = query_processor
        self.context_manager = context_manager
        self.cache_manager = cache_manager
        self.config = config or {}
        
        # Dependency graph for tracking question relationships
        self.dependency_graph = nx.DiGraph()
        
        # Optional separate dependency tracker (if not using query_processor)
        self.dependency_tracker = DependencyTracker(
            config=self.config.get("dependency_tracker_config", {})
        )
        
        # Maximum batch size - adjust based on model context window
        self.max_batch_size = self.config.get("max_batch_size", 10)
        
        # Maximum token budget for batch context
        self.max_batch_token_budget = self.config.get("max_batch_token_budget", 100000)
        
        # Maximum questions to process in parallel
        self.max_parallel_questions = self.config.get("max_parallel_questions", 5)
        
        # Context sharing strategies
        self.context_sharing_threshold = self.config.get("context_sharing_threshold", 0.7)
        
        # Batch job results and metrics
        self.batch_results = {}
        self.metrics = BatchMetrics()
        self.current_batch_id = None
        
        # Token estimation parameters (approximate)
        self.token_estimator = {
            "chars_per_token": 4,  # Approximate characters per token
            "avg_chunk_tokens": 500,  # Average tokens per context chunk
        }

    def process_batch(
        self, 
        questions: List[str],
        available_chunks: List[Chunk] = None,
        batch_id: str = None
    ) -> Tuple[List[Dict[str, Any]], BatchMetrics]:
        """
        Process a batch of questions efficiently.
        
        Args:
            questions: List of questions to process
            available_chunks: Context chunks available for answering
            batch_id: Optional batch identifier
            
        Returns:
            Tuple of (results list, batch metrics)
        """
        start_time = time.time()
        logger.info(f"Starting batch processing of {len(questions)} questions")
        
        # Generate batch ID if not provided
        self.current_batch_id = batch_id or f"batch_{int(time.time())}"
        
        # Initialize metrics
        self.metrics = BatchMetrics()
        self.metrics.total_questions = len(questions)
        
        # Build the dependency graph
        self._build_dependency_graph(questions)
        
        # Analyze the dependency graph to find optimal batching
        batches = self._create_optimal_batches(questions)
        self.metrics.batches_created = len(batches)
        
        # Process each batch
        all_results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} questions")
            
            # If batch has only independent questions, process in parallel
            if self._is_independent_batch(batch):
                batch_results = self._process_independent_batch(batch, available_chunks)
            else:
                # Process sequentially for dependent questions
                batch_results = self._process_dependent_batch(batch, available_chunks)
                
            all_results.extend(batch_results)
        
        # Calculate final metrics
        self.metrics.total_time = time.time() - start_time
        if self.metrics.successful_questions > 0:
            self.metrics.avg_response_time = self.metrics.total_time / self.metrics.successful_questions
        
        # Record metrics and save to cache
        metrics_dict = self.metrics.to_dict()
        self.cache_manager.set(
            f"batch_metrics_{self.current_batch_id}",
            metrics_dict,
            cache_type="metrics",
            ttl=86400 * 7  # Store for 7 days
        )
        
        logger.info(f"Batch processing complete. Tokens saved: {self.metrics.tokens_saved}")
        
        # Before returning results, ensure metrics are consistent
        # Fix the successful/failed count discrepancy
        self.metrics.successful_questions = len([r for r in all_results if r.get('success', False) == True])
        self.metrics.failed_questions = len([r for r in all_results if r.get('success', False) == False])
        
        # Ensure total equals successful + failed
        if self.metrics.successful_questions + self.metrics.failed_questions != self.metrics.total_questions:
            logger.warning(f"Metrics inconsistency detected: total={self.metrics.total_questions}, successful={self.metrics.successful_questions}, failed={self.metrics.failed_questions}")
            # Force consistency
            self.metrics.total_questions = self.metrics.successful_questions + self.metrics.failed_questions
        
        return all_results, self.metrics
    
    def _build_dependency_graph(self, questions: List[str]) -> None:
        """
        Build a dependency graph for the questions.
        
        Args:
            questions: List of questions
        """
        # Add all questions as nodes
        for i, question in enumerate(questions):
            self.dependency_graph.add_node(i, question=question, processed=False)
        
        # Check for dependencies between questions
        for i, current_q in enumerate(questions):
            # Skip first question as it can't have dependencies
            if i == 0:
                continue
                
            # Check previous questions for dependencies
            for j in range(i):
                prev_q = questions[j]
                
                # Check if current question depends on previous
                query_info = self.query_processor.process_query(current_q)
                
                # If there are dependencies, add edges
                if query_info.get("dependencies", {}).get("has_dependencies", False):
                    # Add directed edge from dependency to current question
                    self.dependency_graph.add_edge(j, i, weight=query_info["dependencies"].get("combined_confidence", 0.5))
                    self.metrics.dependent_questions += 1
                    break  # Only record the strongest dependency
        
        # Count independent questions
        self.metrics.independent_questions = self.metrics.total_questions - self.metrics.dependent_questions
        
    def _create_optimal_batches(self, questions: List[str]) -> List[List[str]]:
        """
        Create optimal batches based on dependencies.
        
        Args:
            questions: Original list of questions
            
        Returns:
            List of batches (each batch is a list of question indices)
        """
        # Find connected components (questions with interdependencies)
        components = list(nx.weakly_connected_components(self.dependency_graph))
        
        batches = []
        
        for component in components:
            component_nodes = sorted(list(component))  # Sort to maintain order
            
            # If component is small enough, keep it as one batch
            if len(component_nodes) <= self.max_batch_size:
                batches.append([questions[i] for i in component_nodes])
            else:
                # Split large components into smaller batches
                # For connected questions, we need to maintain their dependency order
                
                # Get topological sort to respect dependencies
                try:
                    # Create subgraph for this component
                    subgraph = self.dependency_graph.subgraph(component_nodes)
                    
                    # Topological sort ensures dependencies are processed first
                    ordered_nodes = list(nx.topological_sort(subgraph))
                    
                    # Split into batches of max_batch_size
                    for i in range(0, len(ordered_nodes), self.max_batch_size):
                        batch_indices = ordered_nodes[i:i+self.max_batch_size]
                        batches.append([questions[i] for i in batch_indices])
                except nx.NetworkXUnfeasible:
                    # Graph has cycles, use different approach
                    logger.warning("Dependency graph has cycles, using sequential processing")
                    # Just process in original order, respecting max batch size
                    for i in range(0, len(component_nodes), self.max_batch_size):
                        batch_indices = component_nodes[i:i+self.max_batch_size]
                        batches.append([questions[i] for i in batch_indices])
        
        return batches
    
    def _is_independent_batch(self, batch: List[str]) -> bool:
        """
        Check if a batch contains only independent questions.
        
        Args:
            batch: List of questions
            
        Returns:
            True if all questions are independent
        """
        # Get indices of these questions in the original list
        question_indices = []
        for question in batch:
            for node in self.dependency_graph.nodes:
                if self.dependency_graph.nodes[node]["question"] == question:
                    question_indices.append(node)
                    break
        
        # Check if there are any edges between these questions
        subgraph = self.dependency_graph.subgraph(question_indices)
        return subgraph.number_of_edges() == 0
    
    def _process_independent_batch(
        self, 
        batch: List[str],
        available_chunks: List[Chunk] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of independent questions in parallel.
        
        Args:
            batch: List of questions
            available_chunks: Available context chunks
            
        Returns:
            List of results
        """
        results = []
        
        # Cap the parallel processing to max_parallel_questions
        max_parallel = min(len(batch), self.max_parallel_questions)
        
        # Use a thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self._process_single_question, q, available_chunks): q 
                for q in batch
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.get("success", False):
                        self.metrics.successful_questions += 1
                    else:
                        self.metrics.failed_questions += 1
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    results.append({
                        "question": question,
                        "answer": f"Error: {str(e)}",
                        "success": False,
                        "error": str(e)
                    })
                    self.metrics.failed_questions += 1
        
        return results
    
    def _process_dependent_batch(
        self, 
        batch: List[str],
        available_chunks: List[Chunk] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of dependent questions sequentially.
        
        Args:
            batch: List of questions
            available_chunks: Available context chunks
            
        Returns:
            List of results
        """
        results = []
        
        # Process in order to respect dependencies
        for question in batch:
            result = self._process_single_question(question, available_chunks)
            results.append(result)
            
            if result.get("success", False):
                self.metrics.successful_questions += 1
            else:
                self.metrics.failed_questions += 1
        
        return results
    
    def _process_single_question(
        self, 
        question: str,
        available_chunks: List[Chunk] = None
    ) -> Dict[str, Any]:
        """
        Process a single question with caching.
        
        Args:
            question: The question to process
            available_chunks: Available context chunks
            
        Returns:
            Result dictionary
        """
        # Check if this question is already in cache
        cache_key = self.cache_manager._generate_key(question)
        cached_result = self.cache_manager.get(cache_key, cache_type="response")
        
        if cached_result:
            # Update metrics for cache hit
            self.metrics.cache_hits += 1
            
            # Check if the cached result already has tokens_saved information
            if "tokens_saved" in cached_result:
                tokens_saved = cached_result["tokens_saved"]
            else:
                # Estimate tokens saved
                tokens_saved = self._estimate_tokens_saved(cached_result)
                # Update the cache with the tokens_saved information for future use
                cached_result["tokens_saved"] = tokens_saved
                self.cache_manager.set(cache_key, cached_result, cache_type="response")
                
            # Update metrics
            self.metrics.tokens_saved += tokens_saved
            
            # Add cache hit info if not already present
            if "cache_hit" not in cached_result:
                cached_result["cache_hit"] = True
            
            # Log with more detailed information
            model_name = getattr(self.api_client, 'model_name', 'Unknown')
            logger.info(f"Cache hit for question: '{question[:50]}...' - Saved ~{tokens_saved} tokens (using {model_name})")
            return cached_result
        
        # Process the query and get context
        query_info = self.query_processor.process_query(question)
        
        # Add available chunks to query_info
        if available_chunks:
            query_info["available_chunks"] = available_chunks
        
        # Get relevant context chunks
        context_chunks = self.context_manager.get_context(
            query=question,
            query_info=query_info
        )
        
        try:
            # Call API to get response using updated API client
            response = self.api_client.generate_response(
                question=question,
                context_chunks=context_chunks,
                query_info=query_info
            )
            
            # Check if response failed
            if not response.get("success", False):
                logger.warning(f"API response failed for question: '{question[:50]}...' - {response.get('error', 'Unknown error')}")
                self.metrics.failed_questions += 1
                return response
            
            # Update success counter
            self.metrics.successful_questions += 1
            
            # Update API call counter
            self.metrics.api_calls += 1
            
            # Store context chunks in response for better token estimation later
            response["context_chunks"] = [chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk for chunk in context_chunks]
            
            # Check if the API response includes token usage information
            if "tokens_used" in response:
                tokens_used = response["tokens_used"]
            else:
                # Calculate tokens used (approximate)
                tokens_used = self._estimate_tokens_used(question, context_chunks, response)
                response["tokens_used"] = tokens_used
                
            self.metrics.tokens_used += tokens_used
            
            # Cache the result
            self.cache_manager.set(
                cache_key,
                response,
                cache_type="response"
            )
            
            # Update conversation history for context
            self.context_manager.update_history(
                question=question,
                response=response,
                query_info=query_info
            )
            
            # Add timestamp to track batch processing
            response["batch_id"] = self.current_batch_id
            response["question"] = question
            
            # Log token usage with model information
            model_name = getattr(self.api_client, 'model_name', 'Unknown')
            logger.info(f"Processed question: '{question[:50]}...' - Used ~{tokens_used} tokens with {model_name}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            self.metrics.failed_questions += 1
            return {
                "success": False,
                "error": error_msg,
                "question": question,
                "batch_id": self.current_batch_id
            }
    
    def _estimate_tokens_saved(self, cached_result: Dict[str, Any]) -> int:
        """
        Estimate tokens saved by using a cached result.
        
        Args:
            cached_result: The cached result
            
        Returns:
            Estimated tokens saved
        """
        # If the cached result already has tokens_saved information, use it
        if "tokens_saved" in cached_result:
            return cached_result["tokens_saved"]
        
        # Get model specific parameters
        model_name = getattr(self.api_client, 'model_name', '')
        
        # Adjust token estimation based on the model
        if "gemini-2.0-flash-thinking-exp-01-21" in model_name:
            # Higher efficiency with Gemini 2.0 Flash Thinking
            chars_per_token = 3.5  # More efficient tokenization
            avg_chunk_tokens = 600  # Higher average token per chunk
            api_overhead = 700  # Higher overhead for the advanced model
            buffer_multiplier = 1.3  # 30% buffer for advanced processing
        else:
            # Standard estimation for other models
            chars_per_token = self.token_estimator["chars_per_token"]
            avg_chunk_tokens = self.token_estimator["avg_chunk_tokens"]
            api_overhead = 500  # Standard overhead
            buffer_multiplier = 1.2  # 20% buffer
            
        # Check if context_chunks information is available
        if "context_chunks" in cached_result and isinstance(cached_result["context_chunks"], list):
            context_chunks = cached_result["context_chunks"]
            # Calculate sum of text lengths in chunks
            context_tokens = 0
            for chunk in context_chunks:
                if isinstance(chunk, dict) and "text" in chunk:
                    context_tokens += len(chunk["text"]) // chars_per_token
                else:
                    context_tokens += avg_chunk_tokens  # Fallback to average if we can't calculate
            
            # Add a small overhead for each chunk (metadata, processing)
            context_tokens += len(context_chunks) * 50
        else:
            # Estimate based on average if detailed info not available
            context_chunks_count = cached_result.get("context_chunks_count", 1)
            context_tokens = context_chunks_count * avg_chunk_tokens
            
        # Question tokens
        question = cached_result.get("question", "")
        question_tokens = len(question) // chars_per_token
        
        # Add response tokens
        response_text = cached_result.get("answer", "")
        response_tokens = len(response_text) // chars_per_token
        
        # API overhead (system prompt, instructions, etc.)
        api_overhead_tokens = api_overhead
        
        # Total tokens saved
        tokens_saved = question_tokens + context_tokens + response_tokens + api_overhead_tokens
        
        # Add buffer for model processing
        return int(tokens_saved * buffer_multiplier)
    
    def _estimate_tokens_used(
        self, 
        question: str, 
        chunks: List[Chunk], 
        response: Dict[str, Any]
    ) -> int:
        """
        Estimate the number of tokens used for a request-response.
        
        Args:
            question: The question text
            chunks: Context chunks used
            response: The response from the API
            
        Returns:
            Estimated token count
        """
        # Question tokens
        question_tokens = len(question) // self.token_estimator["chars_per_token"]
        
        # Context tokens
        context_tokens = sum(len(chunk.text) // self.token_estimator["chars_per_token"] for chunk in chunks)
        
        # Response tokens
        response_text = response.get("answer", "")
        response_tokens = len(response_text) // self.token_estimator["chars_per_token"]
        
        return question_tokens + context_tokens + response_tokens
    
    def visualize_dependency_graph(self, output_path: str = None) -> str:
        """
        Visualize the dependency graph of questions.
        
        Args:
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        if not output_path:
            os.makedirs("static/graphs", exist_ok=True)
            output_path = f"static/graphs/dependency_graph_{self.current_batch_id}.png"
        
        plt.figure(figsize=(12, 8))
        
        # Get positions for nodes
        pos = nx.spring_layout(self.dependency_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.dependency_graph, 
            pos, 
            node_size=700, 
            node_color="skyblue"
        )
        
        # Draw edges with weights as width
        edges = self.dependency_graph.edges(data=True)
        edge_widths = [data["weight"] * 3 for _, _, data in edges]
        nx.draw_networkx_edges(
            self.dependency_graph, 
            pos, 
            width=edge_widths, 
            alpha=0.7, 
            edge_color="gray",
            arrows=True,
            arrowsize=15
        )
        
        # Draw labels - truncate long questions
        labels = {}
        for node in self.dependency_graph.nodes():
            question = self.dependency_graph.nodes[node]["question"]
            labels[node] = f"{node}: {question[:20]}..." if len(question) > 20 else f"{node}: {question}"
        
        nx.draw_networkx_labels(self.dependency_graph, pos, labels, font_size=10)
        
        plt.title("Question Dependency Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Dependency graph visualization saved to {output_path}")
        return output_path
    
    def get_batch_metrics(self, batch_id: str = None) -> Dict[str, Any]:
        """
        Get metrics for a specific batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Metrics dictionary
        """
        batch_id = batch_id or self.current_batch_id
        if not batch_id:
            return self.metrics.to_dict()
            
        # Try to get from cache
        metrics = self.cache_manager.get(
            f"batch_metrics_{batch_id}",
            cache_type="metrics"
        )
        
        return metrics or self.metrics.to_dict()
    
    def save_batch_results(self, results: List[Dict[str, Any]], batch_id: str = None) -> str:
        """
        Save batch results to file.
        
        Args:
            results: List of result dictionaries
            batch_id: Batch identifier
            
        Returns:
            Path to saved file
        """
        batch_id = batch_id or self.current_batch_id or f"batch_{int(time.time())}"
        
        # Create directory if needed
        os.makedirs("cache/batch_results", exist_ok=True)
        
        # Save to file
        output_path = f"cache/batch_results/{batch_id}.json"
        with open(output_path, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics.to_dict(),
                "results": results
            }, f, indent=2)
            
        logger.info(f"Batch results saved to {output_path}")
        return output_path


async def process_batch_async(
    processor: BatchProcessor,
    questions: List[str],
    available_chunks: List[Chunk] = None,
    batch_id: str = None
) -> Tuple[List[Dict[str, Any]], BatchMetrics]:
    """
    Process batch asynchronously (wrapper for non-async implementation).
    
    Args:
        processor: BatchProcessor instance
        questions: List of questions
        available_chunks: Available context chunks
        batch_id: Batch identifier
        
    Returns:
        Tuple of (results list, batch metrics)
    """
    # Run the synchronous method in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: processor.process_batch(questions, available_chunks, batch_id)
    )
