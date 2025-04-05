# Gemini MVP - Batch Transcript Analysis System

## Overview

Gemini MVP is a powerful system for analyzing video transcripts and answering questions using Google's Gemini API. It offers batch processing capabilities, long context handling, and intelligent caching to efficiently process multiple questions against lengthy transcripts. The system supports the full Gemini context window (up to 4M tokens) and implements advanced techniques to optimize token usage and API calls.

## Key Features

- **Long Context Processing**: Handle extensive transcripts (hours of content) using Gemini's 4M token context window
- **Batch Question Processing**: Process multiple related questions efficiently in a single batch
- **Context Caching**: Implement memory and disk caching to reduce API calls and token usage
- **Dependency Tracking**: Identify and leverage relationships between questions for optimal ordering
- **Semantic Chunking**: Break transcripts into meaningful segments that preserve context
- **Interactive UI**: Web-based interface for transcript processing and question answering
- **API Rate Limiting**: Smart rate limiting to prevent throttling and optimize request timing
- **Performance Analytics**: Visualizations and metrics for system performance

## System Architecture

The system consists of several integrated components that work together to provide efficient transcript analysis:

### Core Components

1. **TranscriptProcessor**: Handles parsing and processing of transcript files, manages chunking, and extracts metadata.
2. **BatchProcessor**: Coordinates batch question processing, including dependency detection and optimal ordering.
3. **LongContextHandler**: Optimizes context for lengthy transcripts, ensuring efficient use of token windows.
4. **ContextCache**: Implements caching to store previous interactions and reduce redundancy.
5. **CacheManager**: Manages both memory and disk caching for optimal performance.
6. **QueryProcessor**: Analyzes questions to determine optimal context selection strategies.
7. **ContextManager**: Selects relevant context chunks based on configurable weighting factors.
8. **ApiClient**: Manages communication with the Gemini API, including retries and error handling.
9. **DependencyTracker**: Identifies relationships between questions to optimize processing order.
10. **RateLimiter**: Ensures API requests stay within rate limits and properly spaced.

### Additional Components

- **StreamlitUI**: Web-based interface for interacting with the system.
- **ChunkManager**: Manages text chunks and their metadata from transcripts.
- **Utils**: Helper utilities for various system functions.

## Detailed Process Flow

The system follows a comprehensive workflow from transcript ingestion to answer generation:

### 1. Transcript Processing

- **Input Handling**: Load transcript from text file (with timestamp support)
- **Chunking**: Segment transcript into manageable chunks using either:
  - Fixed-size chunking with configurable overlap
  - Semantic chunking based on topic boundaries
- **Metadata Extraction**: Extract entities, topics, and time references from chunks
- **Chunk Indexing**: Prepare chunks for efficient retrieval during question processing

### 2. Question Analysis

- **Query Classification**: Identify query type (factual, comparative, analysis, etc.)
- **Complexity Assessment**: Determine question complexity to adjust context requirements
- **Entity Recognition**: Extract key entities and terms for context matching
- **Dependency Detection**: Identify relationships to previously asked questions

### 3. Context Selection

- **Relevance Scoring**: Score chunks based on multiple factors:
  - Semantic relevance to the query
  - Entity overlap between chunk and question
  - Temporal recency (for time-based queries)
  - Topic alignment between chunk and question
- **Weight Customization**: Adjust importance of each factor through configurable weights
- **Context Window Optimization**: Select the most relevant chunks within token constraints

### 4. Caching System

- **Two-Tier Caching**: Implement both memory (fast) and disk (persistent) caching
- **Cache Key Generation**: Generate keys based on query and context fingerprints
- **Token Tracking**: Monitor token savings from cache hits
- **TTL Management**: Configure time-to-live for cached entries

### 5. API Interaction

- **Dynamic Prompting**: Select appropriate prompt template based on query type
- **Rate Limiting**: Ensure requests conform to API rate limits (10 RPM for Gemini 2.0)
- **Retry Logic**: Implement exponential backoff for API errors
- **Response Extraction**: Safely extract and validate responses from the API

### 6. Batch Processing

- **Dependency Graph**: Build a graph of question dependencies for optimal ordering
- **Parallel Processing**: Process independent questions in parallel where possible
- **Result Aggregation**: Collect and organize batch results for presentation

### 7. Performance Monitoring

- **Metrics Collection**: Track API calls, token usage, cache performance, and timing
- **Visualization**: Generate charts and graphs for system performance analysis

## Technologies Used

- **Python 3.8+**: Core programming language
- **Google Gemini API**: Large language model for text processing
- **Streamlit**: Web interface framework
- **Pandas & NumPy**: Data processing and analysis
- **Plotly**: Interactive data visualizations
- **Matplotlib**: Static data visualizations
- **NetworkX**: Dependency graph management
- **Regular Expressions**: Pattern matching for transcript parsing

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Google API key with access to Gemini models
- Sufficient API quota for your usage needs

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gemini-mvp.git
   cd gemini-mvp
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```

4. Prepare a transcript file:
   - Plain text format (.txt)
   - Preferably with timestamps in [HH:MM:SS] format
   - One speaker turn per line

### Configuration Options

Create a `config.yaml` file to customize system behavior:

```yaml
# API Configuration
api:
  model: "gemini-2.0-flash-thinking-exp-01-21"
  temperature: 0.2
  max_retries: 3
  retry_delay: 2

# Chunking Configuration
chunking:
  chunk_size: 300
  chunk_overlap: 50
  use_semantic_chunking: true

# Context Selection
context:
  max_chunks: 8
  weights:
    recency: 0.3
    relevance: 0.6
    entity: 0.4
    topic: 0.3
    concept: 0.2

# Caching Configuration
cache:
  disk_cache_dir: "cache"
  memory_cache_size: 100
  ttl: 3600  # Time to live in seconds
```

## Usage Guide

### Command-Line Interface

#### Basic Usage

Process a transcript and answer questions:

```bash
python batch_demo.py --transcript path/to/transcript.txt --questions path/to/questions.json
```

#### Sample Data

Run with provided sample data:

```bash
python batch_demo.py --sample
```

#### Advanced Options

```bash
python batch_demo.py --transcript path/to/transcript.txt \
                    --questions path/to/questions.json \
                    --model gemini-2.0-flash-thinking-exp-01-21 \
                    --output custom_results.json \
                    --api_key your_api_key
```

### Web Interface

Launch the Streamlit UI:

```bash
streamlit run streamlit_ui.py
```

The interface provides:
1. Setup tab for API configuration
2. Transcript processing interface
3. Chunk analysis and visualization
4. Single question interface
5. Batch processing for multiple questions
6. Performance statistics and visualizations

### Input Formats

#### Transcript Format

Plain text file with timestamps (recommended):

```
[00:00:00] Speaker 1: Welcome to our discussion on artificial intelligence.
[00:00:15] Speaker 1: Today we'll explore how AI is transforming industries.
[00:00:30] Speaker 2: Thanks for having me. I'm excited to share insights.
```

#### Questions Format

JSON file with array of questions:

```json
[
  "What are the main topics discussed in this transcript?",
  "What ethical concerns were raised about AI?",
  "What applications of AI were mentioned as promising?",
  "What is predicted for AI technology in the next five years?"
]
```

CSV file with "question" column:

```csv
question
What are the main topics discussed in this transcript?
What ethical concerns were raised about AI?
What applications of AI were mentioned as promising?
What is predicted for AI technology in the next five years?
```

## Performance Optimization

### Token Usage Efficiency

- **Chunk Optimization**: Use semantic chunking to create more meaningful segments
- **Context Selection**: Tune weights to prioritize most relevant chunks
- **Caching**: Implement aggressive caching for repeated or similar questions
- **Dependency Tracking**: Process related questions together to leverage shared context

### API Rate Limiting

The system implements intelligent rate limiting:
- 10 requests per minute (RPM) for Gemini 2.0 Flash
- 60 RPM for Gemini 1.5 models
- 30 RPM for older models
- Minimum spacing between requests (6 seconds for 10 RPM)

### Memory Considerations

For very large transcripts:
- Use disk caching instead of memory caching
- Process transcript in segments if necessary
- Adjust chunk size based on transcript characteristics

## Troubleshooting

### Common Issues

1. **API Authentication Errors**:
   - Ensure your API key is valid and has access to Gemini models
   - Check environment variables are properly set

2. **Rate Limiting Errors**:
   - Check your API quota and usage limits
   - Adjust batch size to process fewer questions at once
   - Increase retry delays in configuration

3. **Memory Issues**:
   - Reduce chunk size for very large transcripts
   - Use disk caching instead of memory caching
   - Process transcript in segments

4. **Empty or Invalid Responses**:
   - Check transcript format and encoding
   - Ensure questions are clear and answerable from the transcript
   - Review API error logs for specific issues

### Logging

The system implements comprehensive logging:

```python
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_mvp.log"),
        logging.StreamHandler()
    ]
)
```

## Advanced Usage

### Custom Prompt Templates

You can customize prompt templates for different query types in the ApiClient class:

```python
# Example custom prompt for analytical questions
templates["analytical"] = """
You are analyzing a transcript of a conversation. Please provide in-depth analysis 
based ONLY on information in the transcript excerpts below.

TRANSCRIPT EXCERPTS:
{context}

ANALYSIS REQUEST: {query}

INSTRUCTIONS:
1. Structure your analysis with clear introduction, key points, and conclusion
2. CITE specific evidence using timestamps [HH:MM:SS] for every analytical point
3. Examine underlying patterns, themes, relationships, and implications
...
"""
```

### Context Weight Tuning

Fine-tune context selection by adjusting weights in the ContextManager:

```python
context_manager = ContextManager(
    config={
        "weights": {
            "recency": 0.3,     # For time-based relevance
            "relevance": 0.6,   # For semantic relevance
            "entity": 0.4,      # For entity matching
            "topic": 0.3,       # For topic alignment
            "concept": 0.2      # For conceptual coverage
        },
        "max_chunks": 8,
        "history_limit": 10
    },
    cache_manager=cache_manager
)
```

### Batch Size Optimization

Adjust batch sizes based on your API quota and performance needs:

```python
results = batch_processor.process_batch(
    questions=questions,
    available_chunks=optimized_chunks,
    batch_id=f"demo_{int(time.time())}",
    batch_size=5  # Adjust based on your needs
)
```

## Contributing

Contributions to improve Gemini MVP are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API team
- Open source libraries used in development

---

Created by Kaushik Sai - 9381046084
