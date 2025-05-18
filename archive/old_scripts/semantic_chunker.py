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

def create_output_directory(output_dir="semantic_chunks"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir

def read_text_file(file_path):
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def analyze_text_structure(text, file_name):
    """Use Gemini to analyze the text structure and identify logical sections."""
    model = genai_client.models.get_model("gemini-1.5-pro-latest")
    
    prompt = f"""
    Analyze the following text from "{file_name}" and identify the logical structure.
    I need you to:
    
    1. Identify the main sections, subsections, and key topics
    2. For each identified section, provide:
       - The level (main section, subsection, etc.)
       - A descriptive title
       - The starting and ending positions in the text (approximate paragraph numbers)
    
    Return your analysis as a structured JSON array with this format:
    [
      {{
        "level": "main_section", 
        "title": "Example Section Title",
        "start_paragraph": 1,
        "end_paragraph": 5,
        "key_concepts": ["concept1", "concept2"]
      }},
      ...
    ]
    
    TEXT TO ANALYZE:
    {text[:15000]}  # Limit text size for API
    
    If the text is truncated, focus on analyzing the provided portion.
    """
    
    try:
        response = model.generate_content(contents=prompt)
        # Extract JSON from the response
        response_text = response.text
        
        # Try to find JSON structure in the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            print("Could not find valid JSON in the response")
            return None
    except Exception as e:
        print(f"Error analyzing text structure: {e}")
        return None

def create_chunks(text, structure, file_name):
    """Create text chunks based on the analyzed structure."""
    if not structure:
        return []
    
    # Split text into paragraphs for easier handling
    paragraphs = text.split('\n\n')
    
    chunks = []
    for section in structure:
        # Extract section information
        level = section.get("level", "unknown")
        title = section.get("title", "Untitled Section")
        start_para = max(0, section.get("start_paragraph", 0) - 1)  # -1 for 0-indexing
        end_para = min(len(paragraphs), section.get("end_paragraph", len(paragraphs)))
        key_concepts = section.get("key_concepts", [])
        
        # Extract the section text
        section_paragraphs = paragraphs[start_para:end_para]
        section_text = '\n\n'.join(section_paragraphs)
        
        # Create chunk with metadata
        chunk = {
            "text": section_text,
            "metadata": {
                "source": file_name,
                "level": level,
                "title": title,
                "key_concepts": key_concepts
            }
        }
        
        chunks.append(chunk)
    
    return chunks

def chunk_large_text(text, file_name, chunk_size=15000):
    """Handle large texts by processing them in chunks."""
    # Split text into reasonable chunks for API processing
    text_chunks = []
    for i in range(0, len(text), chunk_size):
        text_chunks.append(text[i:i+chunk_size])
    
    all_structure = []
    for i, chunk in enumerate(text_chunks):
        print(f"Analyzing chunk {i+1} of {len(text_chunks)}...")
        structure = analyze_text_structure(chunk, f"{file_name} (part {i+1})")
        if structure:
            # Adjust paragraph numbers for chunks after the first
            if i > 0:
                paragraph_count = text[:i*chunk_size].count('\n\n')
                for section in structure:
                    section["start_paragraph"] += paragraph_count
                    section["end_paragraph"] += paragraph_count
            all_structure.extend(structure)
        
        # Respect API rate limits
        if i < len(text_chunks) - 1:
            time.sleep(2)
    
    return all_structure

def process_file(file_path, output_dir):
    """Process a text file into semantic chunks."""
    file_name = Path(file_path).name
    text = read_text_file(file_path)
    
    if not text:
        return
    
    print(f"Processing {file_name}...")
    
    # For large texts, process in chunks
    if len(text) > 15000:
        structure = chunk_large_text(text, file_name)
    else:
        structure = analyze_text_structure(text, file_name)
    
    if not structure:
        print(f"Failed to analyze structure of {file_name}")
        return
    
    # Create chunks based on the analyzed structure
    chunks = create_chunks(text, structure, file_name)
    
    # Save the chunks
    output_file = Path(output_dir) / f"{Path(file_name).stem}_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Created {len(chunks)} semantic chunks for {file_name}")
    print(f"Saved to {output_file}")
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Process text files into semantic chunks using Gemini API")
    parser.add_argument("input_file", help="Text file to process")
    parser.add_argument("--output-dir", default="semantic_chunks", help="Output directory for chunks")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    process_file(args.input_file, output_dir)

if __name__ == "__main__":
    main() 