#!/usr/bin/env python3
"""
Test script to run KSAO analysis on a smaller sample of text to avoid timeouts.
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the Gemini API client
genai_client = genai.Client(api_key=api_key)

def create_output_directory(output_dir="output/test_analysis"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    return output_dir

def get_sample_text(input_file, max_length=10000):
    """
    Get a sample of text from the input file.
    
    Args:
        input_file: Path to the input file
        max_length: Maximum length of the sample text
        
    Returns:
        A sample of text from the input file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sample_text = text[:max_length]
    print(f"Original text length: {len(text)}, Sample text length: {len(sample_text)}")
    return sample_text

def analyze_sample_text(sample_text, output_file):
    """
    Analyze the sample text using Gemini 2.5 Pro to extract KSAOs.
    
    Args:
        sample_text: The sample text to analyze
        output_file: Path to save the analysis results
        
    Returns:
        The analysis text or None if an error occurred
    """
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
    
    I NEED YOU TO DOCUMENT YOUR COMPLETE THINKING PROCESS IN DETAIL:
    1. First, carefully read through the textbook content and note key concepts related to KSAOs
    2. For each section or chapter, identify explicit and implicit competencies
    3. Categorize each KSAO, considering its nature (K, S, A, or O)
    4. Analyze relationships between KSAOs to identify hierarchies and dependencies
    5. Evaluate each KSAO's specificity, malleability, and how it's typically acquired
    6. Organize all findings into a systematic framework
    
    Show your thinking process step-by-step as you analyze the text, including your considerations,
    evaluations, and reasoning. Then present your final findings in a structured format.
    
    NOTE: This is just a small sample of the full textbook. Analyze what's available but indicate areas 
    where more information would be helpful.
    
    TEXTBOOK CONTENT:
    """
    
    print("Sending request to Gemini API...")
    print(f"Text length: {len(sample_text)} characters")
    
    try:
        # First, save the sample text for reference
        with open(f"{output_file}_sample_text.txt", 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        # Send request to Gemini with the latest model
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=prompt + sample_text
        )
        
        # Save the raw response
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Analysis saved to {output_file}")
        return response.text
        
    except Exception as e:
        print(f"Error analyzing sample text: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test KSAO analysis on a sample of text")
    parser.add_argument("--input-file", default="data/gemini_text_output/Chapter_1_full.txt", 
                        help="Input file containing text to analyze")
    parser.add_argument("--output-dir", default="output/test_analysis", 
                        help="Directory for analysis output")
    parser.add_argument("--output-file", default="sample_ksao_analysis.txt", 
                        help="Output file name")
    parser.add_argument("--max-length", type=int, default=10000, 
                        help="Maximum length of the sample text")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    output_file = os.path.join(output_dir, args.output_file)
    
    # Get sample text
    sample_text = get_sample_text(args.input_file, args.max_length)
    
    # Analyze the sample text
    analyze_sample_text(sample_text, output_file)

if __name__ == "__main__":
    main()