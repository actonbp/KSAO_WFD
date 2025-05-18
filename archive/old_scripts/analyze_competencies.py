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

def create_output_directory(output_dir="competency_analysis"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir

def load_chunks(chunks_file):
    """Load semantic chunks from a JSON file."""
    try:
        with open(chunks_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading chunks from {chunks_file}: {e}")
        return None

def analyze_chunk_for_competencies(chunk):
    """Use Gemini to analyze a chunk for competencies and skills."""
    model = genai_client.models.get_model("gemini-1.5-pro-latest")
    
    # Extract the text and metadata from the chunk
    text = chunk.get("text", "")
    metadata = chunk.get("metadata", {})
    title = metadata.get("title", "Untitled")
    
    prompt = f"""
    Analyze the following text about "{title}" and identify the key competencies, skills, and knowledge areas.
    
    For each identified competency or skill, provide:
    1. A clear name/title
    2. A concise description
    3. The type (Knowledge, Skill, Ability, or Other characteristic)
    4. The level of specificity (general or specialized)
    5. Related O*NET occupational categories (if applicable)
    
    Return your analysis as a structured JSON array with this format:
    [
      {{
        "name": "Example Competency Name",
        "description": "Brief description of the competency",
        "type": "Knowledge|Skill|Ability|Other",
        "specificity": "General|Specialized",
        "onet_categories": ["relevant O*NET category"],
        "relevance_score": 0.85  # On a scale of 0-1, how central this is to the text
      }},
      ...
    ]
    
    TEXT TO ANALYZE:
    {text}
    """
    
    try:
        response = model.generate_content(contents=prompt)
        response_text = response.text
        
        # Try to find JSON structure in the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            competencies = json.loads(json_str)
            return competencies
        else:
            print(f"Could not find valid JSON in the response for chunk '{title}'")
            return []
    except Exception as e:
        print(f"Error analyzing competencies for chunk '{title}': {e}")
        return []

def merge_similar_competencies(all_competencies, similarity_threshold=0.7):
    """Merge similar competencies to avoid duplication."""
    model = genai_client.models.get_model("gemini-1.5-pro-latest")
    
    prompt = f"""
    I have a list of competencies and skills extracted from a text. 
    Please merge any that are very similar or redundant, and return a consolidated list.
    
    Use a similarity threshold of {similarity_threshold} (where 1.0 means identical).
    
    For each merged competency, use the name, description, and other attributes from the most comprehensive entry, 
    and include a list of the merged items.
    
    Original competencies:
    {json.dumps(all_competencies, indent=2)}
    
    Return a JSON array of the consolidated competencies with the same structure as the input, 
    but with an additional field "merged_from" for any entries that merged others.
    """
    
    try:
        response = model.generate_content(contents=prompt)
        response_text = response.text
        
        # Try to find JSON structure in the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            merged_competencies = json.loads(json_str)
            return merged_competencies
        else:
            print("Could not find valid JSON in the merge response")
            return all_competencies
    except Exception as e:
        print(f"Error merging competencies: {e}")
        return all_competencies

def analyze_chunks_file(chunks_file, output_dir):
    """Analyze a file containing semantic chunks for competencies."""
    chunks = load_chunks(chunks_file)
    if not chunks:
        return
    
    file_name = Path(chunks_file).stem
    print(f"Analyzing chunks from {file_name}...")
    
    all_competencies = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i+1} of {len(chunks)}...")
        competencies = analyze_chunk_for_competencies(chunk)
        all_competencies.extend(competencies)
        
        # Respect API rate limits
        if i < len(chunks) - 1:
            time.sleep(2)
    
    # Merge similar competencies
    if all_competencies:
        print("Merging similar competencies...")
        merged_competencies = merge_similar_competencies(all_competencies)
        
        # Save the competency analysis
        output_file = Path(output_dir) / f"{file_name}_competencies.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "source_file": chunks_file,
                "competencies": merged_competencies
            }, f, indent=2)
        
        print(f"Identified {len(merged_competencies)} unique competencies/skills")
        print(f"Saved to {output_file}")
        
        return merged_competencies
    
    print("No competencies identified!")
    return []

def main():
    parser = argparse.ArgumentParser(description="Analyze semantic chunks for competencies using Gemini API")
    parser.add_argument("chunks_file", help="JSON file containing semantic chunks")
    parser.add_argument("--output-dir", default="competency_analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    analyze_chunks_file(args.chunks_file, output_dir)

if __name__ == "__main__":
    main() 