#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import time

# Load environment variables
load_dotenv()

# Configure the Gemini API client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the client
genai_client = genai.Client(api_key=api_key)

# Current model version
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"

def create_output_directory(output_dir="output/full_analysis"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    return output_dir

def gather_thinking_traces(input_dir="output/full_analysis"):
    """Gather all thinking traces from chapter analyses."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} not found!")
        return None
    
    # Find all thinking process files
    thinking_files = [f for f in input_path.glob("*_thinking_process.txt")
                     if not "integrated" in f.name]
    
    if not thinking_files:
        print(f"No thinking process files found in {input_dir}")
        return None
    
    print(f"Found {len(thinking_files)} thinking process files")
    
    # Combine all thinking traces with clear separation
    combined_traces = ""
    for file_path in sorted(thinking_files):
        chapter_name = file_path.stem.replace("_ksao_analysis_thinking_process", "")
        print(f"Adding thinking trace for {chapter_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            thinking_text = f.read()
            combined_traces += f"\n\n### THINKING TRACE FOR {chapter_name} ###\n\n{thinking_text}\n\n"
    
    return combined_traces

def analyze_thinking_traces(combined_traces, output_file):
    """
    Use Gemini to analyze thinking traces across chapters and create a summary document.
    
    Args:
        combined_traces: String containing all thinking traces
        output_file: Path to save the analysis
        
    Returns:
        The analysis text or None if an error occurred
    """
    
    prompt = """
    You are an expert in curriculum development, competency mapping, and metacognitive analysis.
    You have been given the thinking traces from an AI system that analyzed different chapters of 
    a substance use disorder (SUD) counselor textbook to identify KSAOs (Knowledge, Skills, Abilities, 
    and Other Characteristics).
    
    Your task is to analyze these thinking traces to create a comprehensive summary document that:
    
    1. Identifies common themes, patterns, and approaches used across the analyses
    2. Highlights key methodological insights about how the AI approached KSAO extraction
    3. Summarizes the reasoning processes that were most effective
    4. Identifies any variations in approach between different chapters and explains why they might occur
    5. Extracts generalizable principles for KSAO identification in professional competency mapping
    
    This meta-analysis should help curriculum developers understand both:
    a) How to effectively identify KSAOs from educational materials
    b) How AI reasoning can be leveraged for competency mapping
    
    Please organize your analysis into clear sections with headings, and provide specific examples 
    from the thinking traces to illustrate your points. Conclude with a set of best practices for 
    KSAO identification and competency mapping.
    
    THINKING TRACES TO ANALYZE:
    """
    
    print("Sending request to Gemini API...")
    print(f"Text length: {len(combined_traces)} characters")
    
    try:
        # First, save the combined traces for reference
        with open(f"{output_file}_source_traces.txt", 'w', encoding='utf-8') as f:
            f.write(combined_traces)
        
        # Send request to Gemini with the latest model and robust error handling
        try:
            print("Sending API request - this may take a few minutes...")
            
            # Add thinking instructions to the prompt
            thinking_prompt = "Think carefully step-by-step to solve this problem. Show your detailed reasoning. Document your thinking process thoroughly.\n\n" + prompt
            
            # Use the simpler API call format
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=thinking_prompt + combined_traces
            )
            print("API request completed successfully")
            if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'thoughts_token_count'):
                print(f"Thoughts token count: {response.usage_metadata.thoughts_token_count}")
                print(f"Total token count: {response.usage_metadata.total_token_count}")
        except Exception as e:
            print(f"API request failed with error: {e}")
            raise
        
        # Save the raw response
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Save a separate file with just the thinking process for this meta-analysis
        thinking_file = f"{output_file}_thinking_process.txt"
        with open(thinking_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Thinking trace analysis saved to {output_file}")
        print(f"Meta-thinking process saved to {thinking_file}")
        return response.text
        
    except Exception as e:
        print(f"Error analyzing thinking traces: {e}")
        # If the text is too long, we might need to split it
        if "content too long" in str(e).lower():
            print("The combined text is too long for the model. Consider processing fewer chapters at once.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze thinking traces across chapter analyses")
    parser.add_argument("--input-dir", default="output/full_analysis", help="Directory containing thinking process files")
    parser.add_argument("--output-dir", default="output/full_analysis", help="Directory for meta-analysis output")
    parser.add_argument("--output-file", default="thinking_trace_analysis.txt", help="Output file name")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    output_file = os.path.join(output_dir, args.output_file)
    
    # Gather all thinking traces
    combined_traces = gather_thinking_traces(args.input_dir)
    
    if combined_traces:
        # Analyze the thinking traces
        analyze_thinking_traces(combined_traces, output_file)

if __name__ == "__main__":
    main()