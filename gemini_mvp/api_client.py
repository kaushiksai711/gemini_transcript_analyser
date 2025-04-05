# gemini_mvp/api_client.py - ACTIVELY USED
# Core component for Gemini API communication, request handling, and response processing

"""
API Client module for communicating with Google's Gemini API.

This module handles API integration, request formatting, error handling,
and implements resilient retry logic.
"""

import os,re
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import random
from datetime import datetime

import google.generativeai as genai
from google.api_core import exceptions, retry
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .chunking import Chunk
from .rate_limiter import RateLimiter
from .dependency_tracker import DependencyTracker

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for interacting with Google's Gemini API.
    
    This class is responsible for:
    1. Formatting requests to the API
    2. Handling API errors and rate limits
    3. Implementing retry logic
    4. Optimizing token usage
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Dict[str, Any] = None):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Google API key (will use GOOGLE_API_KEY env var if not provided)
            config: API configuration
        """
        self.config = config or {}
        
        # Use provided API key or get from environment
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No API key provided. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize Gemini API
        genai.configure(api_key=api_key)
        
        # Configure model
        self.model_name = self.config.get("model", "gemini-2.0-flash-thinking-exp-01-21")
        
        # Set token limits based on model
        if "gemini-2.0-flash-thinking-exp-01-21" in self.model_name:
            self.max_input_tokens = 4000000  # 4M tokens for Gemini 2.0 Flash Thinking
            self.max_output_tokens = 2048
            self.rate_limit_rpm = 10  # 10 requests per minute
        elif "gemini-1.5" in self.model_name:
            self.max_input_tokens = 1000000  # 1M tokens for Gemini 1.5
            self.max_output_tokens = 2048  
            self.rate_limit_rpm = 60  # 60 requests per minute
        else:
            self.max_input_tokens = 32000  # 32k tokens for older models
            self.max_output_tokens = 2048
            self.rate_limit_rpm = 30  # 30 requests per minute
        
        # Configure request parameters
        self.temperature = self.config.get("temperature", 0.2)
        self.max_retries = self.config.get("max_retries", 3)
        self.base_retry_delay = self.config.get("retry_delay", 2)
        self.request_timeout = self.config.get("request_timeout", 60)
        
        # Generation parameters
        self.max_output_tokens = self.config.get("max_output_tokens", 1024)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rpm=self.rate_limit_rpm,
            rpd=1500
        )
        
        # Initialize model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                },
                # safety_settings={
                #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                # }
            )
            logger.info(f"Initialized {self.model_name} model")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            self.model = None
        
        # Initialize dependency tracker
        self.dependency_tracker = DependencyTracker(self.config.get('dependency_tracker_config', {}))
        
        # Initialize result cache
        self.result_cache = {}
        
        # Initialize API call counter
        self.api_calls = 0
        self.total_saved_tokens = 0
    
    def generate_response(
        self,
        question: str,
        context_chunks: List[Chunk],
        query_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a response using the Gemini API.
        
        Args:
            question: The question to answer
            context_chunks: Relevant context chunks
            query_info: Information about the query
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.model:
            return {
                "answer": "Error: Gemini model not initialized.",
                "success": False,
                "error": "Model initialization failed"
            }
        
        # Format prompt with context
        prompt = self._format_prompt(question, context_chunks, query_info)
        
        try:
            # Start timing
            start_time = time.time()
            
            # Make API request with improved error handling
            response_data = self._make_api_request(prompt)
            
            # Check for error
            if "error" in response_data:
                error_msg = response_data["error"]
                logger.error(f"API error: {error_msg}")
                return {
                    "answer": f"Error generating response: {error_msg}",
                    "success": False,
                    "response_time": time.time() - start_time,
                    "error": error_msg
                }
            
            # Get response text
            response_text = response_data["text"]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract timestamps if mentioned in the response
            timestamps = self._extract_timestamps(response_text, context_chunks)
            
            # Calculate confidence based on answer quality indicators
            confidence = self._calculate_confidence(response_text, question, context_chunks)
            
            return {
                "answer": response_text,
                "success": True,
                "response_time": response_time,
                "timestamps": timestamps,
                "confidence": confidence,
                "context_chunks": [chunk.chunk_id for chunk in context_chunks],
                "query_type": query_info.get("query_type", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def get_quota_status(self) -> Dict[str, Any]:
        """
        Get the current quota status for the API.
        
        Returns:
            Dictionary with quota status information
        """
        status = self.rate_limiter.get_quota_status()
        
        # Add model info
        status["model"] = self.model_name
        status["model_limits"] = {
            "rpm": self.rate_limit_rpm,
            "rpd": 1500
        }
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get API client statistics.
        
        Returns:
            Dictionary with API client statistics
        """
        stats = {
            "api_calls": self.api_calls,
            "total_saved_tokens": self.total_saved_tokens,
        }
        
        # Add cache stats if available
        if hasattr(self, "cache_manager") and hasattr(self.cache_manager, "get_stats"):
            cache_stats = self.cache_manager.get_stats()
            if cache_stats:
                stats["hits"] = cache_stats.get("memory_hits", 0) + cache_stats.get("disk_hits", 0)
                stats["misses"] = cache_stats.get("memory_misses", 0) + cache_stats.get("disk_misses", 0)
                stats["hit_ratio"] = cache_stats.get("hit_ratio", 0)
        else:
            stats["hits"] = 0
            stats["misses"] = 0
            stats["hit_ratio"] = 0
            
        return stats
    
    def _format_prompt(
        self,
        question: str,
        context_chunks: List[Chunk],
        query_info: Dict[str, Any]
    ) -> str:
        """
        Format the prompt with context for the API.
        
        Args:
            question: The question to answer
            context_chunks: Relevant context chunks
            query_info: Information about the query
            
        Returns:
            Formatted prompt
        """
        # Format context from chunks with better organization
        context_parts = []
        for chunk in context_chunks:
            time_info = ""
            if chunk.start_time and chunk.end_time:
                time_info = f"[{chunk.start_time} - {chunk.end_time}]"
            
            # Format each chunk with its timestamp range
            chunk_text = f"{time_info}\n{chunk.text}\n"
            context_parts.append(chunk_text)
        
        context_text = "\n\n".join(context_parts)
        
        # Get query type
        query_type = query_info.get("query_type", "unknown")
        
        # Add dependencies if present (for follow-up questions)
        dependencies_text = ""
        dependencies = query_info.get("dependencies", {})
        if dependencies.get("has_dependencies", False) and "referenced_items" in dependencies:
            referenced = dependencies["referenced_items"]
            dependencies_text = "Previous conversation context:\n\n"
            for item in referenced:
                dependencies_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"
        
        # Construct the prompt based on query type
        prompt_template = self._get_prompt_template(query_type)
        
        # Format the prompt
        prompt = prompt_template.format(
            context=context_text,
            query=question,
            dependencies=dependencies_text
        )
        
        return prompt
        
    def _get_prompt_template(self, query_type: str) -> str:
        """
        Get the appropriate prompt template for the query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Prompt template string
        """
        templates = {
            "factual": """
You are analyzing a transcript of a conversation. Please answer the following question based ONLY on information explicitly provided in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question using ONLY information from the provided transcript excerpts
2. If the transcript doesn't explicitly contain the answer, state this clearly rather than speculating
3. Always CITE your sources using timestamps [HH:MM:SS] when referring to specific information
4. Include multiple timestamps from different parts of the transcript if available
5. Be thorough and precise - include all relevant details from the transcript
6. Structure your answer with clear formatting (headings, bullet points) as appropriate
7. If the transcript presents multiple viewpoints, acknowledge all perspectives
8. Do not include any information not found in the provided transcript

Aim to provide a comprehensive, accurate answer that directly addresses the question using only facts from the transcript.
""",
            "summarization": """
You are analyzing a transcript of a conversation. Please summarize the information based ONLY on the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

SUMMARIZATION REQUEST: {query}

INSTRUCTIONS:
1. Provide a concise yet comprehensive summary focused on what was requested
2. Include ONLY information from the provided transcript excerpts
3. Organize the summary in a logical structure with clear headings and sections
4. CITE specific parts using timestamps [HH:MM:SS] throughout your summary
5. Include ALL major points and key details relevant to the request
6. Maintain neutrality - do not add your own opinions or external information
7. Use bullet points or numbered lists for multiple points or sequential information
8. Highlight important connections, patterns or themes that emerge across different parts
9. If the transcript contains conflicting viewpoints, acknowledge all perspectives fairly

Your summary should provide a balanced, thorough overview of what the transcript reveals about the requested topic.
""",
            "comparative": """
You are analyzing a transcript of a conversation. Please compare or contrast elements based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

COMPARISON REQUEST: {query}

INSTRUCTIONS:
1. Structure your response with clear comparative framework (point-by-point, pros/cons, similarities/differences)
2. CITE specific evidence using timestamps [HH:MM:SS] for every comparison point
3. Create a comparison table if appropriate
4. Include ONLY comparisons that can be directly supported by the transcript excerpts
5. Balance your treatment of all elements being compared
6. Note both technical AND non-technical aspects (ethical, social, practical issues)
7. Highlight any proposed solutions or approaches to address these challenges
8. Assess the relative emphasis or priority given to different challenges
9. Identify any tensions or trade-offs between different solutions or approaches
10. Note any unresolved questions or areas of ongoing debate around these challenges
11. If challenges are presented from multiple perspectives, acknowledge all viewpoints

Provide a balanced, evidence-based comparison that thoroughly examines all relevant dimensions mentioned in the transcript.
""",
            "analytical": """
You are analyzing a transcript of a conversation. Please provide in-depth analysis based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

ANALYSIS REQUEST: {query}

INSTRUCTIONS:
1. Structure your analysis with clear introduction, key points, and conclusion
2. CITE specific evidence using timestamps [HH:MM:SS] for every analytical point
3. Examine underlying patterns, themes, relationships, and implications
4. Consider multiple perspectives presented in the transcript
5. Analyze both explicit statements AND implicit assumptions or subtle patterns
6. Assess the strength of evidence for different viewpoints presented
7. Identify any tensions, contradictions, or gaps in the discussion
8. Examine how different factors interact with or influence each other
9. Connect specific details to broader themes or concepts mentioned
10. Note limitations of the analysis based on what information is/isn't in the transcript

Your analysis should be thorough, nuanced, and deeply grounded in the transcript content while showing insight into complex relationships between ideas.
""", 
            "explanatory": """
You are analyzing a transcript of a conversation. Please explain concepts, processes, or ideas based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

EXPLANATION REQUEST: {query}

INSTRUCTIONS:
1. Structure your explanation in a logical flow from basic to more complex aspects
2. CITE specific parts using timestamps [HH:MM:SS] throughout your explanation
3. Define ALL relevant terminology exactly as it's presented in the transcript
4. Break down complex concepts into clear, digestible components
5. Use analogies or examples ONLY if they appear in the transcript itself
6. Explain relationships between different concepts or components
7. Highlight any caveats, limitations, or nuances mentioned about the concept
8. Present multiple perspectives or interpretations if the transcript does so
9. If explanations in the transcript are technical, maintain the appropriate level of detail
10. If the transcript contains incomplete explanations, acknowledge these gaps

Your explanation should be comprehensive, accessible, and accurately reflect how the concept is presented in the transcript.
""",
            "challenges": """
You are analyzing a transcript of a conversation. Please explore challenges, issues, or problems mentioned in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

QUESTION ABOUT CHALLENGES: {query}

INSTRUCTIONS:
1. Identify and describe ALL relevant challenges mentioned in the transcript
2. CITE specific evidence using timestamps [HH:MM:SS] for each challenge discussed
3. Structure your response with clear categories or themes of related challenges
4. Explain the context, causes, and implications of each challenge as presented
5. Note both technical AND non-technical aspects (ethical, social, practical issues)
6. Highlight any proposed solutions or approaches to address these challenges
7. Assess the relative emphasis or priority given to different challenges
8. Identify any tensions or trade-offs between different solutions or approaches
9. Note any unresolved questions or areas of ongoing debate around these challenges
10. If challenges are presented from multiple perspectives, acknowledge all viewpoints

Your response should provide a balanced, thorough exploration of the challenges as they are presented in the transcript.
""",
            "ethical": """
You are analyzing a transcript of a conversation. Please examine ethical considerations or implications discussed in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

ETHICS QUESTION: {query}

INSTRUCTIONS:
1. Identify ALL ethical dimensions, dilemmas, or considerations mentioned in the transcript
2. CITE specific discussions using timestamps [HH:MM:SS] for each ethical point
3. Structure your response around distinct ethical themes or principles mentioned
4. Present multiple perspectives on ethical issues if the transcript contains them
5. Note both explicit ethical discussions AND implicit ethical dimensions
6. Highlight how various stakeholders or interests might be affected differently
7. Identify any tensions between competing ethical values or principles
8. Note any proposed frameworks, guidelines, or approaches for addressing ethical concerns
9. Assess the depth and breadth of ethical consideration in the transcript
10. If consensus on ethical matters is presented, or if disagreement exists, note this

Your response should provide a nuanced, balanced exploration of ethical dimensions exactly as they appear in the transcript.
""",
            "technical": """
You are analyzing a transcript of a conversation. Please provide technical details or specifications based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

TECHNICAL QUESTION: {query}

INSTRUCTIONS:
1. Provide precise technical details exactly as described in the transcript
2. CITE specific technical information using timestamps [HH:MM:SS] throughout
3. Structure your response with appropriate technical categories or components
4. Use technical terminology consistent with how it's used in the transcript
5. Include quantitative data, specifications, or measurements if mentioned
6. Explain technical processes or workflows in the same sequence as presented
7. Note any technical limitations, constraints, or dependencies mentioned
8. Include implementation details, requirements, or prerequisites if discussed
9. If technical alternatives or trade-offs are presented, include all options
10. Present technical information at the same level of depth as the transcript

Your response should be technically precise, comprehensive, and accurately reflect the technical content of the transcript.
""",
            "implementation": """
You are analyzing a transcript of a conversation. Please extract implementation details, examples, or demonstrations based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

IMPLEMENTATION QUESTION: {query}

INSTRUCTIONS:
1. Focus on specific implementation steps, procedures, or examples from the transcript
2. CITE relevant implementation details using timestamps [HH:MM:SS] 
3. Present implementation steps in the correct sequential order
4. Include exact code snippets, commands, or syntax ONLY if specifically mentioned
5. Highlight practical considerations, gotchas, or best practices that are discussed
6. Note any implementation alternatives or variations that are presented
7. Include environment requirements, dependencies, or prerequisites if mentioned
8. Explain how implementation connects to broader concepts or goals discussed
9. If debugging or problem-solving examples are provided, include these details
10. Note any limitations, concerns, or future improvements mentioned regarding implementation

Your response should provide practical, actionable information exactly as presented in the transcript.
""",
            "default": """
You are analyzing a transcript of a conversation. Please answer the following question based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer using ONLY information found in the provided transcript excerpts
2. CITE your sources using timestamps [HH:MM:SS] when referring to specific information
3. If the information isn't in the transcript, clearly state this rather than speculating
4. Be comprehensive - include all relevant details from the transcript
5. Structure your answer with appropriate formatting for readability
6. If the transcript presents multiple viewpoints, acknowledge all perspectives
7. Focus on providing accurate, thorough information directly from the source material

Your goal is to provide a well-organized, accurate, and comprehensive answer based solely on the transcript content.
"""
        }
        
        return templates.get(query_type, templates["default"])
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to select appropriate prompt template.
        
        Args:
            query: The user's query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Check for explicit keywords in the query
        if any(term in query_lower for term in ["summarize", "summary", "overview", "briefing", "main points"]):
            return "summarization"
            
        if any(term in query_lower for term in ["compare", "contrast", "difference", "similarities", "versus", "vs"]):
            return "comparative"
            
        if any(term in query_lower for term in ["analyze", "analysis", "evaluate", "assessment", "implications", "significance"]):
            return "analytical"
            
        if any(term in query_lower for term in ["explain", "how does", "what is", "definition", "describe", "clarify"]):
            return "explanatory"
            
        if any(term in query_lower for term in ["challenge", "problem", "difficulty", "issue", "obstacle", "drawback"]):
            return "challenges"
            
        if any(term in query_lower for term in ["ethics", "ethical", "moral", "right", "wrong", "responsible", "fair", "justice"]):
            return "ethical"
            
        if any(term in query_lower for term in ["technical", "specification", "architecture", "system", "framework", "technology"]):
            return "technical"
            
        if any(term in query_lower for term in ["implement", "code", "example", "demo", "build", "create", "develop", "deployment"]):
            return "implementation"
        
        # Look for question types
        if re.search(r'(what|which|who|when|where)\s', query_lower):
            return "factual"
            
        if re.search(r'why\s|how\s', query_lower):
            return "explanatory"
            
        # Default to factual for simple questions
        return "factual"
    
    def _extract_timestamps(self, response_text: str, context_chunks: List[Dict[str, Any]] = None) -> List[str]:
        """
        Extract timestamps mentioned in the response.
        
        Args:
            response_text: Response text to scan
            context_chunks: Context chunks used for generation
            
        Returns:
            List of extracted timestamps
        """
        import re
        # Match timestamp patterns like [00:15:30] or timestamps without brackets
        timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]|(\d{2}:\d{2}:\d{2}(?:\.\d+)?)'
        
        timestamps = []
        for match in re.finditer(timestamp_pattern, response_text):
            # Get the group that matched (either with or without brackets)
            timestamp = match.group(1) if match.group(1) else match.group(2)
            timestamps.append(timestamp)
            
        # Also check context chunks if provided
        if context_chunks:
            # Extract timestamps from chunk text if available
            for chunk in context_chunks:
                if isinstance(chunk, dict) and 'text' in chunk:
                    chunk_text = chunk['text']
                    # Use finditer instead of findall to extract the same way as above
                    for match in re.finditer(timestamp_pattern, chunk_text):
                        timestamp = match.group(1) if match.group(1) else match.group(2)
                        timestamps.append(timestamp)
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    # Use finditer instead of findall to extract the same way as above
                    for match in re.finditer(timestamp_pattern, chunk_text):
                        timestamp = match.group(1) if match.group(1) else match.group(2)
                        timestamps.append(timestamp)
        
        # Remove duplicates and sort
        timestamps = sorted(list(set(timestamps)))
        
        return timestamps
    
    def _calculate_confidence(self, response_text: str, question: str = None, context_chunks: List[Dict[str, Any]] = None) -> float:
        """
        Calculate a confidence score for the response.
        
        Args:
            response_text: The response text
            question: The original question
            context_chunks: Context chunks used for generation
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base score
        score = 0.5
        
        # Adjust score based on response characteristics
        
        # Length of response (longer responses tend to be more comprehensive)
        if len(response_text) > 500:
            score += 0.1
        elif len(response_text) < 100:
            score -= 0.1
        
        # Presence of timestamps (indicates specific referencing)
        if re.search(r'\[\d{2}:\d{2}:\d{2}\]', response_text):
            score += 0.2
        
        # Certainty markers
        certainty_markers = ['certainly', 'definitely', 'clearly', 'indeed', 'absolutely']
        uncertainty_markers = ['perhaps', 'maybe', 'might', 'could be', 'possibly', 'I\'m not sure']
        
        for marker in certainty_markers:
            if marker.lower() in response_text.lower():
                score += 0.05
                
        for marker in uncertainty_markers:
            if marker.lower() in response_text.lower():
                score -= 0.05
        
        # Formatting quality
        if '\n' in response_text and len(response_text.split('\n')) > 3:
            score += 0.1  # Well-structured with paragraphs
            
        # Check if specific question terms are addressed in the response
        if question:
            key_terms = [term.lower() for term in question.split() if len(term) > 3]
            matched_terms = sum(1 for term in key_terms if term.lower() in response_text.lower())
            term_coverage = matched_terms / max(1, len(key_terms))
            score += term_coverage * 0.2
            
        # Ensure score is between 0 and 1
        return max(0.1, min(1.0, score))
    
    def _make_api_request(self, prompt: str, attempt: int = 1) -> Dict[str, Any]:
        """
        Make an API request to the Gemini API.
        
        Args:
            prompt: The prompt to send
            attempt: The current attempt number
            
        Returns:
            API response
        """
        try:
            # Wait for rate limiter
            wait_time = self.rate_limiter.wait_for_token()
            if wait_time > 0:
                logger.info(f"Rate limited, waiting for {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            # Log attempt
            logger.info(f"Making API request (attempt {attempt}/{self.max_retries})")
            
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Make the API request
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            # Check explicitly for empty candidates first
            if not hasattr(response, 'candidates') or not response.candidates:
                logger.warning("API returned empty candidates")
                
                # Retry if we have attempts left
                if attempt < self.max_retries:
                    logger.info(f"Retrying API request (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(self.base_retry_delay * attempt)  # Exponential backoff
                    return self._make_api_request(prompt, attempt + 1)
                
                # Max retries reached, return error
                return {
                    "error": "API returned empty response after all retry attempts",
                    "success": False
                }
            
            # Process successful response
            self.api_calls += 1  # Increment API call counter
            
            # Extract response text safely - handle the exact error from logs
            try:
                # First try the simple accessor
                response_text = response.text
                return {"text": response_text, "response": response, "success": True}
            except Exception as e:
                logger.warning(f"Could not access response.text: {e}")
                
                # The error message suggests response.candidates is empty in some cases
                # Let's handle this specific case
                if "response.candidates is empty" in str(e):
                    # This might mean content filtering triggered or another API issue
                    if attempt < self.max_retries:
                        logger.info(f"Empty candidates detected, retrying (attempt {attempt+1}/{self.max_retries})")
                        time.sleep(self.base_retry_delay * attempt)
                        return self._make_api_request(prompt, attempt + 1)
                    else:
                        return {
                            "error": "Content filtering may have triggered or API returned empty candidates",
                            "success": False
                        }
                
                # Try direct access to candidates structure
                try:
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                return {"text": candidate.content.parts[0].text, "response": response, "success": True}
                except Exception as e2:
                    logger.error(f"Failed to access candidate structure: {e2}")
                
                # If we reached here, we failed to extract text after all attempts
                if attempt < self.max_retries:
                    logger.info(f"Retrying after extraction error (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(self.base_retry_delay * attempt)
                    return self._make_api_request(prompt, attempt + 1)
                
                return {
                    "error": f"Error extracting text: {e}",
                    "success": False
                }
        
        except Exception as e:
            logger.error(f"API request error: {e}")
            
            # Check if we should retry
            if attempt < self.max_retries:
                retry_delay = self.base_retry_delay * attempt  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                return self._make_api_request(prompt, attempt + 1)
            
            # Max retries reached, return error
            return {"error": str(e), "success": False}

    def process_batch(self, questions: List[str], context_provider, dependencies: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of questions against the given context provider.
        
        Args:
            questions: List of questions to process
            context_provider: Object that provides context for questions
            dependencies: Optional explicit dependencies data
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        # If we have an existing session, use dependency tracker
        if self.dependency_tracker.question_history and dependencies is None:
            # Check for dependencies between questions in the batch
            batch_dependencies = {}
            for i, question in enumerate(questions):
                deps = self.dependency_tracker.find_dependencies(question)
                if deps:
                    batch_dependencies[i] = deps
            
            # Process questions with dependencies first, in proper order
            sorted_indices = self._determine_question_order(questions, batch_dependencies)
            ordered_questions = [questions[i] for i in sorted_indices]
            
            # Process questions in the determined order
            for idx, question_idx in enumerate(sorted_indices):
                question = questions[question_idx]
                
                # Get dependencies for current question
                question_deps = batch_dependencies.get(question_idx, [])
                
                # Process the question with dependencies
                response = self.process_question(
                    question, 
                    context_provider[question_idx],
                    dependencies=[dep for dep in question_deps if dep["composite_score"] > 0.7]
                )
                
                # Add to results at the original position
                results.append(response)
                
                # Add to dependency tracker for future questions
                self.dependency_tracker.add_to_history(
                    question, 
                    response.get("answer", ""), 
                    response.get("metadata", {})
                )
        else:
            # Process questions without considering dependencies
            for question in questions:
                response = self.process_question(question, context_provider, dependencies=dependencies)
                results.append(response)
                
                # Still add to dependency tracker for future batches
                self.dependency_tracker.add_to_history(
                    question, 
                    response.get("answer", ""), 
                    response.get("metadata", {})
                )
        
        return results
    
    def _determine_question_order(self, questions: List[str], dependencies_map: Dict[int, List[Dict[str, Any]]]) -> List[int]:
        """
        Determine optimal processing order for questions based on dependencies.
        
        Args:
            questions: List of questions
            dependencies_map: Map of question index to dependencies
            
        Returns:
            List of question indices in processing order
        """
        # Create a directed graph representing dependencies
        graph = {i: [] for i in range(len(questions))}
        
        # For each question with dependencies
        for q_idx, deps in dependencies_map.items():
            for dep in deps:
                # Find which questions in the current batch match this dependency
                dep_question = dep["question"]
                for other_idx, other_q in enumerate(questions):
                    # Skip self-dependencies
                    if other_idx == q_idx:
                        continue
                        
                    # If this question matches a dependency, add edge
                    if other_q == dep_question:
                        graph[q_idx].append(other_idx)
        
        # Perform topological sort to determine order
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                # Cycle detected, process in arbitrary order
                return
            if node in visited:
                return
                
            temp_visited.add(node)
            
            # Visit dependencies first
            for neighbor in graph[node]:
                visit(neighbor)
                
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        # Visit all nodes
        for i in range(len(questions)):
            if i not in visited:
                visit(i)
                
        # Reverse to get correct order
        return result[::-1]
    
    def process_question(self, question: str, context_provider, dependencies: Optional[List[Dict[str, Any]]] = None, query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single question using the Gemini API.
        
        Args:
            question: The question to process
            context_provider: Object that provides context for questions
            dependencies: Optional dependencies from previous QA pairs
            query_type: Type of query
            
        Returns:
            Dictionary with answer and metadata
        """
        # Check cache first
        cache_key = f"question_{hash(question)}"
        if cache_key in self.result_cache:
            logger.info(f"Using cached result for: {question}")
            
            # Estimate tokens saved by using cache
            cached_result = self.result_cache[cache_key]
            
            # Estimate tokens in prompt
            prompt_tokens = self._estimate_tokens(question)
            if isinstance(context_provider, str):
                prompt_tokens += self._estimate_tokens(context_provider)
            
            # Estimate tokens in response
            response_tokens = self._estimate_tokens(cached_result.get("answer", ""))
            
            # Update token savings counter
            total_tokens_saved = prompt_tokens + response_tokens
            self.total_saved_tokens += total_tokens_saved
            
            logger.info(f"Saved approximately {total_tokens_saved} tokens by using cache")
            
            return self.result_cache[cache_key]
        
        # Format prompt with context
        prompt = self._build_prompt(question, context_provider, dependencies, query_type)
        
        try:
            # Start timing
            start_time = time.time()
            
            # Make API request with improved error handling
            response_data = self._make_api_request(prompt)
            
            # Check for error
            if "error" in response_data:
                error_msg = response_data["error"]
                logger.error(f"API error: {error_msg}")
                return {
                    "question": question,
                    "answer": f"Error generating response: {error_msg}",
                    "success": False,
                    "response_time": time.time() - start_time,
                    "error": error_msg
                }
            
            # Get response text
            response_text = response_data["text"]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract timestamps if mentioned in the response
            timestamps = self._extract_timestamps(response_text)
            
            # Calculate confidence based on answer quality indicators
            confidence = self._calculate_confidence(response_text, question)
            
            return {
                "question": question,
                "answer": response_text,
                "success": True,
                "response_time": response_time,
                "timestamps": timestamps,
                "confidence": confidence,
                "query_type": query_type,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "question": question,
                "answer": f"Error generating response: {str(e)}",
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _build_prompt(self, question: str, context_provider, dependencies: Optional[List[Dict[str, Any]]] = None, query_type: Optional[str] = None) -> str:
        """
        Build a prompt for the Gemini API.
        
        Args:
            question: Question to answer
            context_provider: Object that provides context for questions
            dependencies: Optional dependencies from previous QA pairs
            query_type: Type of query
            
        Returns:
            Formatted prompt
        """
        # Detect query type if not provided
        if query_type is None:
            query_type = self._detect_query_type(question)
        # Append dependency information
        dependencies_text = ""
        if dependencies:
            dependencies_text = "PREVIOUS RELATED INTERACTIONS:\n"
            for item in dependencies:
                dependencies_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"
                
        # Construct the prompt based on query type
        prompt_template = self._get_prompt_template(query_type)
        
        # Format the prompt
        prompt = prompt_template.format(
            context=context_provider,
            query=question,
            dependencies=dependencies_text
        )
        
        return prompt

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation based on word count
        # In practice, you'd want to use the tokenizer from the model
        word_count = len(text.split())
        return int(word_count * 1.3)  # Rough approximation

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate token count for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: 1 token ≈ 4 characters
        return len(text) // 4
