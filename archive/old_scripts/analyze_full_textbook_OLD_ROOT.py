#!/usr/bin/env python3
import os
import json
import argparse
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

def create_output_directory(output_dir="full_analysis"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir

def gather_all_chapter_text(input_dir="gemini_text_output"):
    """Gather text from all chapter files."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} not found!")
        return None
    
    # Find all full text files
    full_text_files = sorted(list(input_path.glob("*_full.txt")))
    
    if not full_text_files:
        print(f"No chapter files found in {input_dir}")
        return None
    
    print(f"Found {len(full_text_files)} chapter files")
    
    # Combine all chapter texts
    combined_text = ""
    for file_path in full_text_files:
        chapter_name = file_path.stem.replace("_full", "")
        print(f"Adding {chapter_name} to combined text...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()
            combined_text += f"\n\n### {chapter_name} ###\n\n{chapter_text}"
    
    return combined_text

def analyze_full_textbook(combined_text, output_file):
    """Analyze the full textbook text using Gemini."""
    # No need to get model, use it directly in the API call
    
    prompt = """
    You are an expert in curriculum development and competency mapping for substance use disorder (SUD) counselors. 
    You need to analyze the following textbook content to identify the complete set of Knowledge, Skills, Abilities, 
    and Other Characteristics (KSAOs) required for SUD counselors.
    
    For each identified KSAO, provide:
    1. A clear name/title
    2. A complete description
    3. The classification (Knowledge, Skill, Ability, or Other characteristic)
    4. The specificity level (general or specialized)
    5. Related O*NET occupational categories
    6. Stability/malleability classification (whether this KSAO is relatively fixed or can be developed)
    7. Explicit/tacit orientation (whether this is explicitly taught or tacitly acquired)
    8. Prerequisites or developmental relationships (what must be learned before this)
    
    Additionally, identify any hierarchical structure among these KSAOs:
    - Which KSAOs represent dimensions vs. sub-dimensions
    - How KSAOs relate to each other in terms of development sequence
    - Which KSAOs serve as foundations for others
    
    PROVIDE YOUR DETAILED THINKING PROCESS as you analyze the text. Then organize your final findings
    in a well-structured format.
    
    TEXTBOOK CONTENT:
    """
    
    print("Sending request to Gemini API...")
    print(f"Text length: {len(combined_text)} characters")
    
    try:
        # First, save the combined text for reference
        with open(f"{output_file}_combined_text.txt", 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Send request to Gemini with the latest model
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=prompt + combined_text
        )
        
        # Save the raw response
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Analysis saved to {output_file}")
        return response.text
        
    except Exception as e:
        print(f"Error analyzing textbook: {e}")
        # If the text is too long, we might need to split it
        if "content too long" in str(e).lower():
            print("The combined text is too long for the model. Consider using the chunking approach instead.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze full textbook content for SUD counselor KSAOs")
    parser.add_argument("--input-dir", default="gemini_text_output", help="Directory containing chapter text files")
    parser.add_argument("--output-dir", default="full_analysis", help="Directory for analysis output")
    parser.add_argument("--output-file", default="textbook_ksao_analysis.txt", help="Output file name")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    output_file = os.path.join(output_dir, args.output_file)
    
    # Gather all chapter text
    combined_text = gather_all_chapter_text(args.input_dir)
    
    if combined_text:
        # Analyze the combined text
        analyze_full_textbook(combined_text, output_file)

if __name__ == "__main__":
    main() 