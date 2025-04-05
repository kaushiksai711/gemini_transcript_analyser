"""
Demo script for the Gemini Video Transcript Analysis MVP.

This script demonstrates how to use the library to process a transcript
and ask questions about it.
"""
GOOGLE_API_KEY=""
import os
import time
from gemini_mvp import TranscriptProcessor
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
def main():
    """Run the demo."""
    # Print welcome message
    print("=" * 80)
    print("Gemini Video Transcript Analysis MVP - Demo")
    print("=" * 80)
    
    # Check for API key
    #api_key = os.environ.get("GOOGLE_API_KEY")
    api_key=GOOGLE_API_KEY
    if not api_key:
        print("WARNING: No API key found. Set the GOOGLE_API_KEY environment variable.")
        print("You can get an API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize processor
    print("\nInitializing transcript processor...")
    processor = TranscriptProcessor(
        config_path="../config.yaml",
        api_key=api_key,
        verbose=True
    )
    
    # Load transcript
    print("\nLoading sample transcript...")
    transcript_path = "D:\google_gemini_mvp\samples\sample_transcript.txt"
    
    if not os.path.exists(transcript_path):
        print(f"Error: Could not find transcript file at {transcript_path}")
        return
        
    processor.load_transcript_from_file(transcript_path)
    print(f"Processed transcript into {processor.chunk_manager.chunk_count} chunks")
    # for chunk in processor.chunk_manager.chunks:
    #     print(f"Chunk {chunk.chunk_id}: {chunk.text}")
    
    # Ask some questions
    # demo_questions = [
    #     "What is artificial intelligence?",
    #     "What are the different types of AI mentioned?",
    #     "How is AI used in healthcare?",
    #     "What challenges related to AI were discussed?",
    #     "Can you tell me more about the bias issue mentioned earlier?"
    # ]
    demo_questions = [
        "What type of habitat is described as being home to a variety of wildlife,including beavers and grass snakes?",
"Why were beavers considered extinct in Britain, and what has contributed to their reemergence?",
"Describe the symbiotic relationship between reed warblers and cuckoos as mentioned in the text.",
"What role do human activities play in supporting or endangering wildlife in British urban areas, according to the text?",
"Analyze the impact of reintroduced species like the great bustard on the ecosystems of Britain, highlighting both challenges and benefits."
    ]
    
    print("\nAsking demo questions...")
    for i, question in enumerate(demo_questions):
        time.sleep(8)
        print(f"\nQuestion {i+1}: {question}")
        
        # Track time for response
        start_time = time.time()
        response = processor.ask(question)
        elapsed = time.time() - start_time
        
        # Print answer
        print(f"Answer ({elapsed:.2f}s):")
        print(response.get("answer", ""))
        
        # Print metadata
        confidence = response.get("confidence", 0)
        timestamps = response.get("timestamps", [])
        
        if timestamps:
            print(f"\nMentioned timestamps: {', '.join(timestamps)}")
        
        print(f"Confidence: {confidence:.2f}")
        
        # Brief pause between questions
        if i < len(demo_questions) - 1:
            time.sleep(1)
    
    # Export session
    session_file = "demo_session.json"
    print(f"\nExporting session to {session_file}...")
    processor.export_session(session_file)
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Try the CLI with: python -m gemini_mvp process -t samples/sample_transcript.txt")
    print("2. Start interactive mode: python -m gemini_mvp interactive")
    print("3. Check out the README.md for more usage options")

if __name__ == "__main__":
    main()
