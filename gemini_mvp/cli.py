# gemini_mvp/cli.py - NOT ACTIVELY USED in batch_demo.py
# Command-line interface for the package

"""
Command Line Interface for Gemini Video Transcript Analysis MVP.

This module provides a CLI for processing transcripts and asking questions.
"""

import os
import sys
import time
import logging
import click
import yaml
import json
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import colorama
from colorama import Fore, Style

from .transcript_processor import TranscriptProcessor
from .utils import setup_logging, generate_timestamp_str, load_config

# Initialize colorama
colorama.init()

# Logger for CLI
logger = logging.getLogger(__name__)

# Create a global context for passing data between commands
class Context:
    def __init__(self):
        self.processor = None
        self.config_path = "config.yaml"
        self.verbose = False
        self.cache_dir = None
        self.output_format = "text"

pass_context = click.make_pass_decorator(Context, ensure=True)

@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--cache-dir', help='Directory for persistent cache')
@click.option('--output-format', '-o', type=click.Choice(['text', 'json', 'yaml']), 
              default='text', help='Output format')
@click.version_option(package_name='gemini_mvp')
@pass_context
def cli(ctx, config, verbose, cache_dir, output_format):
    """Gemini Video Transcript Analysis MVP - Process video transcripts and ask questions."""
    # Initialize colorama
    colorama.init()
    
    # Set context values
    ctx.config_path = config or ctx.config_path
    ctx.verbose = verbose
    ctx.cache_dir = cache_dir
    ctx.output_format = output_format
    
    # Setup logging
    setup_logging(verbose)
    
    # Welcome message
    if sys.stdout.isatty():
        click.echo(f"{Fore.CYAN}Gemini Video Transcript Analysis MVP{Style.RESET_ALL}")
        click.echo(f"{Fore.CYAN}================================={Style.RESET_ALL}")

@cli.command()
@click.option('--transcript', '-t', required=True, help='Path to transcript file')
@click.option('--question', '-q', help='Question to answer')
@click.option('--api-key', help='Google API key (or set GOOGLE_API_KEY environment variable)')
@pass_context
def process(ctx, transcript, question, api_key):
    """Process a transcript and optionally ask a question."""
    # Initialize processor
    _ensure_processor(ctx, transcript, api_key)
    
    click.echo(f"{Fore.GREEN}Transcript processed successfully!{Style.RESET_ALL}")
    click.echo(f"Created {ctx.processor.chunk_manager.chunk_count} chunks")
    
    # Ask question if provided
    if question:
        ask_question(ctx, question)

@cli.command()
@click.option('--question', '-q', required=True, help='Question to answer')
@pass_context
def ask(ctx, question):
    """Ask a question about the processed transcript."""
    if not ctx.processor or not ctx.processor.transcript_loaded:
        click.echo(f"{Fore.RED}No transcript loaded. Use 'process' command first.{Style.RESET_ALL}")
        return
    
    ask_question(ctx, question)

@cli.command()
@click.option('--transcript', '-t', required=True, help='Path to transcript file')
@click.option('--questions', '-q', required=True, help='Path to questions file (one per line)')
@click.option('--output', '-o', help='Path to save results')
@click.option('--api-key', help='Google API key (or set GOOGLE_API_KEY environment variable)')
@pass_context
def batch(ctx, transcript, questions, output, api_key):
    """Process multiple questions from a file."""
    # Initialize processor
    _ensure_processor(ctx, transcript, api_key)
    
    # Read questions
    try:
        with open(questions, 'r', encoding='utf-8') as f:
            question_list = [line.strip() for line in f if line.strip()]
    except Exception as e:
        click.echo(f"{Fore.RED}Error reading questions file: {e}{Style.RESET_ALL}")
        return
    
    click.echo(f"Processing {len(question_list)} questions...")
    
    # Process questions
    responses = []
    for i, question in enumerate(tqdm(question_list, desc="Questions")):
        response = ctx.processor.ask(question)
        responses.append({
            "question": question,
            "answer": response.get("answer", ""),
            "confidence": response.get("confidence", 0),
            "timestamps": response.get("timestamps", [])
        })
    
    # Save results if output specified
    if output:
        try:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=2)
            click.echo(f"{Fore.GREEN}Results saved to {output}{Style.RESET_ALL}")
        except Exception as e:
            click.echo(f"{Fore.RED}Error saving results: {e}{Style.RESET_ALL}")
    
    # Print summary
    click.echo(f"{Fore.GREEN}Processed {len(responses)} questions{Style.RESET_ALL}")

@cli.command()
@click.option('--file', '-f', required=True, help='Path to save session')
@pass_context
def save(ctx, file):
    """Save the current session to a file."""
    if not ctx.processor or not ctx.processor.transcript_loaded:
        click.echo(f"{Fore.RED}No active session to save.{Style.RESET_ALL}")
        return
    
    success = ctx.processor.export_session(file)
    if success:
        click.echo(f"{Fore.GREEN}Session saved to {file}{Style.RESET_ALL}")
    else:
        click.echo(f"{Fore.RED}Failed to save session.{Style.RESET_ALL}")

@cli.command()
@click.option('--file', '-f', required=True, help='Path to session file')
@pass_context
def load(ctx, file):
    """Load a previously saved session."""
    # Initialize processor if needed
    if not ctx.processor:
        config = load_config(ctx.config_path)
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        ctx.processor = TranscriptProcessor(
            config_path=ctx.config_path,
            api_key=api_key,
            cache_dir=ctx.cache_dir,
            verbose=ctx.verbose
        )
    
    success = ctx.processor.import_session(file)
    if success:
        click.echo(f"{Fore.GREEN}Session loaded from {file}{Style.RESET_ALL}")
    else:
        click.echo(f"{Fore.RED}Failed to load session.{Style.RESET_ALL}")

@cli.command()
@pass_context
def reset(ctx):
    """Reset conversation history."""
    if not ctx.processor:
        click.echo(f"{Fore.RED}No active session to reset.{Style.RESET_ALL}")
        return
    
    ctx.processor.reset_conversation()
    click.echo(f"{Fore.GREEN}Conversation history reset.{Style.RESET_ALL}")

@cli.command()
@pass_context
def interactive(ctx):
    """Start interactive question-answering mode."""
    if not ctx.processor or not ctx.processor.transcript_loaded:
        click.echo(f"{Fore.RED}No transcript loaded. Use 'process' command first.{Style.RESET_ALL}")
        return
    
    click.echo(f"{Fore.CYAN}Starting interactive mode. Type 'exit' to quit.{Style.RESET_ALL}")
    click.echo(f"{Fore.CYAN}Type 'reset' to clear conversation history.{Style.RESET_ALL}")
    
    while True:
        # Get user input
        question = input(f"{Fore.YELLOW}Question: {Style.RESET_ALL}")
        question = question.strip()
        
        # Check for exit command
        if question.lower() in ('exit', 'quit'):
            break
            
        # Check for reset command
        if question.lower() == 'reset':
            ctx.processor.reset_conversation()
            click.echo(f"{Fore.GREEN}Conversation history reset.{Style.RESET_ALL}")
            continue
            
        # Skip empty input
        if not question:
            continue
            
        # Process question
        ask_question(ctx, question)

def ask_question(ctx, question):
    """Process a question and display the answer."""
    click.echo(f"Question: {question}")
    
    # Get start time for response time tracking
    start_time = time.time()
    
    # Show spinner while processing
    with tqdm(total=0, desc="Processing", bar_format="{desc}: {elapsed}s") as pbar:
        # Process question
        response = ctx.processor.ask(question)
        
        # Update progress
        processing_time = time.time() - start_time
        pbar.set_description(f"Processed in {processing_time:.2f}s")
    
    # Display answer based on output format
    if ctx.output_format == 'json':
        click.echo(json.dumps(response, indent=2))
    elif ctx.output_format == 'yaml':
        click.echo(yaml.dump(response, sort_keys=False))
    else:
        # Text format
        answer = response.get("answer", "")
        confidence = response.get("confidence", 0)
        timestamps = response.get("timestamps", [])
        
        # Display answer with styling
        click.echo(f"{Fore.GREEN}Answer:{Style.RESET_ALL}")
        click.echo(answer)
        
        # Display metadata
        if timestamps:
            click.echo(f"\n{Fore.CYAN}Mentioned timestamps:{Style.RESET_ALL} {', '.join(timestamps)}")
        
        click.echo(f"{Fore.CYAN}Confidence:{Style.RESET_ALL} {confidence:.2f}")
        click.echo(f"{Fore.CYAN}Response time:{Style.RESET_ALL} {response.get('response_time', 0):.2f}s")

def _ensure_processor(ctx, transcript_path, api_key=None):
    """Initialize processor if needed."""
    if ctx.processor and ctx.processor.transcript_loaded:
        return
    
    # Initialize processor
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    
    # Show spinner while loading
    with tqdm(total=0, desc="Initializing", bar_format="{desc}: {elapsed}s") as pbar:
        ctx.processor = TranscriptProcessor(
            config_path=ctx.config_path,
            api_key=api_key,
            cache_dir=ctx.cache_dir,
            verbose=ctx.verbose
        )
        pbar.set_description("Loading transcript")
        ctx.processor.load_transcript_from_file(transcript_path)

if __name__ == '__main__':
    cli()
