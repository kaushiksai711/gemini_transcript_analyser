"""
Batch Prediction Demo with Long Context and Context Caching

This script showcases the enhanced Gemini MVP system with:
1. Efficient batch prediction capabilities
2. Long context handling and optimization
3. Advanced context caching for token savings
4. Dependency tracking between related questions

Usage:
    python batch_demo.py --transcript path/to/transcript.txt --questions path/to/questions.json
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Gemini MVP modules
from gemini_mvp.transcript_processor import TranscriptProcessor
from gemini_mvp.batch_processor import BatchProcessor
from gemini_mvp.long_context_handler import LongContextHandler
from gemini_mvp.context_cache import ContextCache
from gemini_mvp.cache_manager import CacheManager
from gemini_mvp.api_client import GeminiClient


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch Prediction Demo with Gemini API')
    
    parser.add_argument('--transcript', type=str, required=False,
                      help='Path to transcript file')
    parser.add_argument('--questions', type=str, required=False,
                      help='Path to questions JSON file')
    parser.add_argument('--api_key', type=str, required=False,
                      help='Google API key (defaults to GOOGLE_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash-thinking-exp-01-21',
                      help='Model to use (default: gemini-2.0-flash-thinking-exp-01-21)')
    parser.add_argument('--output', type=str, default='batch_results.json',
                      help='Output file for batch results')
    parser.add_argument('--sample', action='store_true',
                      help='Use sample transcript and questions')
    
    return parser.parse_args()


def load_or_create_sample_transcript():
    """Load sample transcript or create one if it doesn't exist."""
    sample_path = Path("samples/10hrs.txt")
    
    if sample_path.exists():
        logger.info(f"Loading sample transcript from {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            return f.read()
    
    # Create a sample transcript
    logger.info("Creating sample transcript")
#     sample_transcript = """[00:00:00] Speaker 1: Welcome to our discussion on artificial intelligence and its impact on society.
# [00:00:15] Speaker 1: Today we'll explore how AI is transforming various industries and the ethical implications that come with these advancements.
# [00:00:30] Speaker 2: Thanks for having me. I'm excited to share insights from both academic research and industry applications.
# [00:01:00] Speaker 1: Let's start with a basic question: How would you define artificial intelligence for our audience?
# [00:01:15] Speaker 2: Artificial intelligence refers to computer systems designed to perform tasks that typically require human intelligence.
# [00:01:30] Speaker 2: These include learning from experience, recognizing patterns, understanding language, and making decisions.
# [00:02:00] Speaker 2: It's important to distinguish between narrow AI, which focuses on specific tasks, and general AI, which would have human-like cognitive abilities across domains.
# [00:02:30] Speaker 1: What are some of the most significant recent advancements in AI technology?
# [00:03:00] Speaker 2: Large language models like GPT-4 and Gemini have revolutionized natural language processing and generation.
# [00:03:15] Speaker 2: These models can write essays, summarize documents, translate languages, and even generate creative content with remarkable fluency.
# [00:03:45] Speaker 2: Computer vision has also made tremendous progress, with systems now able to identify objects, recognize faces, and interpret visual scenes with high accuracy.
# [00:04:15] Speaker 2: And reinforcement learning has enabled breakthroughs in robotics, game playing, and autonomous systems.
# [00:04:45] Speaker 1: How are these technologies being applied in real-world industries?
# [00:05:00] Speaker 2: In healthcare, AI is improving diagnostic accuracy, drug discovery, and personalized treatment plans.
# [00:05:30] Speaker 2: For example, AI systems can analyze medical images to detect cancer at early stages with accuracy comparable to specialized radiologists.
# [00:06:00] Speaker 2: In finance, AI algorithms handle fraud detection, risk assessment, and automated trading strategies.
# [00:06:30] Speaker 2: Manufacturing uses AI for quality control, predictive maintenance, and optimizing production lines.
# [00:07:00] Speaker 2: And we're seeing AI transform customer service through chatbots and virtual assistants that can handle increasingly complex interactions.
# [00:07:30] Speaker 1: What about the ethical concerns associated with AI deployment?
# [00:08:00] Speaker 2: Bias and fairness are major concerns. AI systems trained on biased data will reproduce and potentially amplify those biases in their decisions.
# [00:08:30] Speaker 2: Privacy is another critical issue, especially with the vast amounts of personal data needed to train effective AI systems.
# [00:09:00] Speaker 2: There's also the question of accountability - when AI systems make mistakes or cause harm, who is responsible?
# [00:09:30] Speaker 2: And perhaps most concerning long-term is the potential for job displacement as AI automates tasks previously performed by humans.
# [00:10:00] Speaker 1: How should we approach regulating AI to address these concerns?
# [00:10:30] Speaker 2: We need a multi-faceted approach involving both industry self-regulation and governmental oversight.
# [00:11:00] Speaker 2: Transparency requirements are essential - organizations should disclose when AI is being used and explain how decisions are made.
# [00:11:30] Speaker 2: Impact assessments should be conducted before deploying AI in high-risk applications like healthcare or criminal justice.
# [00:12:00] Speaker 2: And we need comprehensive data protection laws that give individuals control over how their personal information is used.
# [00:12:30] Speaker 1: What about AI safety research? How are scientists addressing risks from more advanced systems?
# [00:13:00] Speaker 2: Alignment research focuses on ensuring AI systems act in accordance with human values and intentions.
# [00:13:30] Speaker 2: Researchers are developing techniques to make AI systems interpretable, so we can understand their decision-making processes.
# [00:14:00] Speaker 2: Robustness research aims to create systems that behave predictably even in novel situations.
# [00:14:30] Speaker 2: And there's significant work on establishing proper governance structures for overseeing the development of more powerful AI systems.
# [00:15:00] Speaker 1: Looking toward the future, what developments do you anticipate in the next decade?
# [00:15:30] Speaker 2: We'll likely see AI systems with stronger reasoning capabilities and better common sense understanding.
# [00:16:00] Speaker 2: Multimodal AI that can process and generate text, images, audio, and video simultaneously will become more sophisticated.
# [00:16:30] Speaker 2: Human-AI collaboration tools will transform knowledge work across many professions.
# [00:17:00] Speaker 2: And I expect significant advances in AI for scientific discovery, particularly in fields like materials science and drug development.
# [00:17:30] Speaker 1: What advice would you give to organizations looking to implement AI responsibly?
# [00:18:00] Speaker 2: Start by clearly defining the problem you're trying to solve and assess whether AI is truly the appropriate solution.
# [00:18:30] Speaker 2: Invest in diverse teams to develop and oversee AI systems, including experts in ethics and relevant domain knowledge.
# [00:19:00] Speaker 2: Implement robust testing procedures before deployment, including adversarial testing to identify potential weaknesses.
# [00:19:30] Speaker 2: And establish ongoing monitoring systems to track performance and address issues that emerge when the AI is operating in the real world.
# [00:20:00] Speaker 1: And for individuals concerned about AI's impact on their careers?
# [00:20:30] Speaker 2: Focus on developing skills that complement AI rather than compete with it - creativity, emotional intelligence, ethical reasoning, and interpersonal communication.
# [00:21:00] Speaker 2: Embrace continuous learning to adapt to changing technological landscapes.
# [00:21:30] Speaker 2: And look for opportunities to use AI as a tool to enhance your own capabilities and productivity.
# [00:22:00] Speaker 1: Thank you for these insights. To conclude, what's your overall perspective on AI's role in society?
# [00:22:30] Speaker 2: I'm cautiously optimistic. AI has extraordinary potential to address major challenges in healthcare, climate science, education, and beyond.
# [00:23:00] Speaker 2: But realizing this potential while minimizing harms requires thoughtful design, appropriate governance, and inclusive conversations about how these technologies should be developed.
# [00:23:30] Speaker 2: The goal should be creating AI that amplifies human capabilities, respects human autonomy, and contributes to a more equitable and sustainable world.
# [00:24:00] Speaker 1: That's a powerful vision. Thank you for sharing your expertise with us today.
# [00:24:15] Speaker 2: It's been my pleasure. I look forward to continuing this important conversation.
# """
    
    # Ensure the directory exists
    os.makedirs(sample_path.parent, exist_ok=True)
    
    # Save the sample transcript
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_transcript)
    
    return sample_transcript


def load_or_create_sample_questions():
    """Load sample questions or create them if they don't exist."""
    sample_path = Path("samples/10hrs_questions.json")
    
    if sample_path.exists():
        logger.info(f"Loading sample questions from {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Create sample questions
    logger.info("Creating sample questions")
    sample_questions = [
  "What makes lotus silk so rare and labor-intensive, and how is it extracted by hand?",
  "How do Japanese chef’s knives compare in craftsmanship and price to the lotus silk featured earlier?",
  "What shared values of precision and tradition are observed between championship chess sets and Japanese calligraphy ink?",
  "Why are gooseneck barnacles considered a delicacy, and how does their harvesting reflect the same rarity as lotus silk?",
  "What materials and geographic factors contribute to the cost of Olympic curling stones?",
  "How does the making of Japanese swords demonstrate both cultural heritage and modern luxury, similar to Japanese calligraphy ink?",
  "In what ways does the creation of Hasselblad cameras reflect a similar blend of precision and brand legacy as Steinway pianos?",
  "Why are Japanese Ruby Roman grapes considered a luxury, and how does their cultivation mirror that of the Nonthaburi durian?",
  "How does the sourcing and aging process of sandalwood compare to that of Zellige tiles in terms of craftsmanship and origin?",
  "What final insights does the video offer about the intersection of culture, tradition, and value across all luxury items featured?"
]

    
    # Ensure the directory exists
    os.makedirs(sample_path.parent, exist_ok=True)
    
    # Save the sample questions
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_questions, f, indent=2)
    
    return sample_questions


def visualize_batch_results(results, metrics, output_dir="static/graphs"):
    """Visualize batch processing results and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize token savings - handle case where both might be zero
    plt.figure(figsize=(10, 6))
    labels = ['Tokens Saved', 'Tokens Used']
    
    # Ensure we have positive values for the pie chart to avoid "Invalid vertices array" error
    tokens_saved = max(1, metrics.tokens_saved) if hasattr(metrics, 'tokens_saved') else 1
    tokens_used = max(1, metrics.tokens_used) if hasattr(metrics, 'tokens_used') else 1
    
    # If both values are minimal, use placeholder values for visualization
    if tokens_saved <= 1 and tokens_used <= 1:
        sizes = [100, 100]  # Equal parts as placeholder
    else:
        sizes = [tokens_saved, tokens_used]
    
    colors = ['green', 'gray']
    explode = (0.1, 0)  # explode the 1st slice (Tokens Saved)
    
    # Use try-except to catch any visualization errors
    try:
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Token Usage Efficiency')
        plt.savefig(f"{output_dir}/token_savings.png", dpi=300, bbox_inches="tight")
    except Exception as e:
        logger.warning(f"Error creating token savings pie chart: {e}")
        # Fall back to a bar chart which is more reliable
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=colors)
        plt.title('Token Usage Efficiency')
        plt.ylabel('Tokens')
        plt.savefig(f"{output_dir}/token_savings.png", dpi=300, bbox_inches="tight")
    
    plt.close()
    
    # Visualize success rates - ensure positive values
    plt.figure(figsize=(10, 6))
    categories = ['Successful', 'Failed', 'Cache Hits']
    
    # Safely get values with defaults
    successful = max(0, getattr(metrics, 'successful_questions', 0))
    failed = max(0, getattr(metrics, 'failed_questions', 0))
    cache_hits = max(0, getattr(metrics, 'cache_hits', 0))
    
    values = [successful, failed, cache_hits]
    colors = ['blue', 'red', 'green']
    
    plt.bar(categories, values, color=colors)
    plt.title('Question Processing Results')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/success_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Visualize API call savings - ensure positive values
    plt.figure(figsize=(10, 6))
    api_labels = ['API Calls Made', 'API Calls Saved']
    
    # Safely get values with defaults
    api_calls = max(0, getattr(metrics, 'api_calls', 0))
    total_questions = max(0, getattr(metrics, 'total_questions', 0))
    api_calls_saved = max(0, total_questions - api_calls)
    
    api_values = [api_calls, api_calls_saved]
    api_colors = ['orange', 'green']
    
    plt.bar(api_labels, api_values, color=api_colors)
    plt.title('API Call Efficiency')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/api_calls.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")
    return output_dir


def main():
    """Run the batch prediction demo."""
    args = parse_args()
    
    # Load transcript
    if args.transcript:
        logger.info(f"Loading transcript from {args.transcript}")
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript_text = f.read()
    elif args.sample:
        transcript_text = load_or_create_sample_transcript()
    else:
        # Try to use uploaded_transcript.txt if it exists
        transcript_path = Path("uploaded_transcript.txt")
        if transcript_path.exists():
            logger.info(f"Loading transcript from {transcript_path}")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        else:
            transcript_text = load_or_create_sample_transcript()
    
    # Load questions
    if args.questions:
        logger.info(f"Loading questions from {args.questions}")
        with open(args.questions, "r", encoding="utf-8") as f:
            questions = json.load(f)
    elif args.sample:
        questions = load_or_create_sample_questions()
    else:
        questions = load_or_create_sample_questions()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        logger.error("No API key provided. Set GOOGLE_API_KEY environment variable or use --api_key.")
        return
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Create cache manager
    cache_manager = CacheManager({
        "disk_cache_dir": "cache",
        "memory_cache_size": 100
    })
    
    # Initialize API client
    api_client = GeminiClient(
        api_key=api_key,
        config={"model": args.model}
    )
    
    # Create transcript processor
    processor = TranscriptProcessor(
        transcript_text=transcript_text,
        api_key=api_key,
        verbose=True
    )
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        api_client=api_client,
        query_processor=processor.query_processor,
        context_manager=processor.context_manager,
        cache_manager=cache_manager
    )
    
    # Initialize long context handler
    long_context_handler = LongContextHandler(
        chunk_manager=processor.chunk_manager,
        cache_manager=cache_manager
    )
    
    # Initialize context cache
    context_cache = ContextCache(
        cache_manager=cache_manager
    )
    
    # Process transcript for long context handling
    logger.info("Processing transcript for long context handling...")
    long_context_handler.process_transcript(transcript_text)
    
    # Find optimal context for the batch
    logger.info("Finding optimal context for question batch...")
    optimized_chunks = long_context_handler.optimize_context_for_batch(questions=questions)
    
    # Process the batch
    logger.info(f"Processing {len(questions)} questions...")
    start_time = time.time()
    
    batch_results, metrics = batch_processor.process_batch(
        questions=questions,
        available_chunks=optimized_chunks,
        batch_id=f"demo_{int(time.time())}"
    )
    
    processing_time = time.time() - start_time
    logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
    
    # Generate visualization
    logger.info("Generating visualizations...")
    viz_dir = visualize_batch_results(batch_results, metrics)
    
    # Create dependency graph visualization
    dep_graph_path = batch_processor.visualize_dependency_graph()
    logger.info(f"Dependency graph saved to {dep_graph_path}")
    
    # Save results
    output_path = args.output
    logger.info(f"Saving results to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": batch_results,
            "metrics": metrics.to_dict(),
            "processing_time": processing_time,
            "timestamp": time.time(),
            "config": {
                "model": args.model,
                "questions_count": len(questions),
                "transcript_length": len(transcript_text)
            },
            "visualizations": {
                "dependency_graph": dep_graph_path,
                "token_savings": f"{viz_dir}/token_savings.png",
                "success_rates": f"{viz_dir}/success_rates.png",
                "api_calls": f"{viz_dir}/api_calls.png"
            }
        }, f, indent=2)
    
    # Display summary results
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total questions processed: {metrics.total_questions}")
    print(f"Successful questions: {metrics.successful_questions}")
    print(f"Failed questions: {metrics.failed_questions}")
    print(f"Cache hits: {metrics.cache_hits}")
    print(f"API calls made: {metrics.api_calls}")
    print(f"API calls saved: {metrics.total_questions - metrics.api_calls}")
    print(f"Token savings: {metrics.tokens_saved:,} tokens")
    print(f"Token savings percentage: {(metrics.tokens_saved / max(1, metrics.tokens_saved + metrics.tokens_used)) * 100:.1f}%")
    print(f"Average response time: {metrics.avg_response_time:.2f} seconds")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    print(f"Visualizations saved to: {viz_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
