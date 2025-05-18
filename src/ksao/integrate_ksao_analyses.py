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
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"  # Using same model as other scripts

def create_output_directory(output_dir="output/full_analysis"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    return output_dir

def gather_chapter_analyses(input_dir="output/full_analysis"):
    """Gather all individual chapter KSAO analyses."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} not found!")
        return None
    
    # Find all KSAO analysis files (excluding thinking_process and combined_text files)
    analysis_files = [f for f in input_path.glob("*_ksao_analysis.txt") 
                      if not f.name.endswith("_thinking_process.txt") 
                      and not f.name.endswith("_combined_text.txt")
                      and not "integrated" in f.name]
    
    if not analysis_files:
        print(f"No chapter analysis files found in {input_dir}")
        return None
    
    print(f"Found {len(analysis_files)} chapter analysis files")
    
    # Combine all chapter analyses with clear separation
    combined_analyses = ""
    for file_path in sorted(analysis_files):
        chapter_name = file_path.stem.replace("_ksao_analysis", "")
        print(f"Adding analysis for {chapter_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            analysis_text = f.read()
            combined_analyses += f"\n\n### ANALYSIS FOR {chapter_name} ###\n\n{analysis_text}\n\n"
    
    return combined_analyses

def integrate_ksao_analyses(combined_analyses, output_file):
    """
    Use Gemini to integrate separate chapter KSAO analyses into a cohesive framework.
    
    Args:
        combined_analyses: String containing all chapter analyses
        output_file: Path to save the integrated analysis
        
    Returns:
        The integrated analysis text or None if an error occurred
    """
    
    prompt = """
    You are an expert in curriculum development and competency mapping for substance use disorder (SUD) counselors.
    You have been given separate KSAO (Knowledge, Skills, Abilities, and Other Characteristics) analyses for 
    different chapters of a SUD counselor textbook.
    
    Your task is to integrate these separate chapter analyses into ONE COMPREHENSIVE KSAO FRAMEWORK that 
    covers the entire textbook. This should be a well-organized, cohesive document that eliminates redundancies,
    resolves inconsistencies, and presents a clear hierarchical organization of KSAOs.
    
    For each identified KSAO in your integrated framework, provide:
    1. A clear name/title
    2. A complete description that synthesizes information across chapters
    3. The classification (Knowledge, Skill, Ability, or Other characteristic)
    4. The specificity level (general or specialized)
    5. Related O*NET occupational categories
    6. Stability/malleability classification (whether this KSAO is relatively fixed or can be developed)
    7. Explicit/tacit orientation (whether this is explicitly taught or tacitly acquired)
    8. Prerequisites or developmental relationships (what must be learned before this)
    
    Additionally, identify the hierarchical structure among these KSAOs:
    - Which KSAOs represent dimensions vs. sub-dimensions
    - How KSAOs relate to each other in terms of development sequence
    - Which KSAOs serve as foundations for others
    
    I NEED YOU TO DOCUMENT YOUR COMPLETE THINKING PROCESS IN DETAIL:
    1. First, review all chapter analyses to identify common themes, redundancies, and unique elements
    2. Then categorize and group related KSAOs across chapters
    3. Resolve any inconsistencies in how similar KSAOs are classified or described
    4. Create a cohesive framework that logically organizes all KSAOs
    5. Identify the relationships and hierarchies among KSAOs
    
    Show your thinking process step-by-step as you integrate the analyses, including your considerations,
    evaluations, and reasoning. Then present your final integrated framework in a structured format.
    
    CHAPTER ANALYSES TO INTEGRATE:
    """
    
    print("Sending request to Gemini API...")
    print(f"Text length: {len(combined_analyses)} characters")
    
    try:
        # First, save the combined analyses for reference
        with open(f"{output_file}_source_analyses.txt", 'w', encoding='utf-8') as f:
            f.write(combined_analyses)
        
        # Send request to Gemini with the latest model and robust error handling
        try:
            print("Sending API request - this may take a few minutes...")
            
            # Add thinking instructions to the prompt
            thinking_prompt = "Think carefully step-by-step to solve this problem. Show your detailed reasoning. Document your thinking process thoroughly.\n\n" + prompt
            
            # Use the simpler API call format
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=thinking_prompt + combined_analyses
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
        
        # Save a separate file with just the thinking process
        thinking_file = f"{output_file}_thinking_process.txt"
        with open(thinking_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Integrated analysis saved to {output_file}")
        print(f"Thinking process saved to {thinking_file}")
        return response.text
        
    except Exception as e:
        print(f"Error integrating analyses: {e}")
        # If the text is too long, we might need to split it
        if "content too long" in str(e).lower():
            print("The combined text is too long for the model. Consider processing fewer chapters at once.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Integrate separate chapter KSAO analyses into a cohesive framework")
    parser.add_argument("--input-dir", default="output/full_analysis", help="Directory containing chapter KSAO analyses")
    parser.add_argument("--output-dir", default="output/full_analysis", help="Directory for integrated output")
    parser.add_argument("--output-file", default="integrated_ksao_framework.txt", help="Output file name")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    output_file = os.path.join(output_dir, args.output_file)
    
    # Gather all chapter analyses
    combined_analyses = gather_chapter_analyses(args.input_dir)
    
    if combined_analyses:
        # Integrate the analyses
        integrate_ksao_analyses(combined_analyses, output_file)

if __name__ == "__main__":
    main()