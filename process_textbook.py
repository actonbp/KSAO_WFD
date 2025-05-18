#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import time
import json
import glob
import io
import base64
from PIL import Image

# Load environment variables from .env file (contains GEMINI_API_KEY)
load_dotenv()

# Get the API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the Gemini API client
client = genai.Client(api_key=api_key)

# Current model version
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"  # Using Gemini 2.5 Pro for best quality

# Create output directory if it doesn't exist
def create_output_directory(output_dir):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    return output_dir

# Convert an image to base64 for API consumption
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Process a single page and extract text using Gemini API
def extract_text_from_page(img):
    """
    Extract text from a single image using the Gemini 2.5 Pro API.
    
    Args:
        img: PIL Image object containing the page to process
        
    Returns:
        String containing the extracted text
    """
    # Convert the image to base64
    base64_image = image_to_base64(img)
    
    # Create the request content with the image
    contents = {
        "parts": [
            {
                "inline_data": {
                    "mime_type": "image/png", 
                    "data": base64_image
                }
            },
            {
                "text": "Extract all text from this image. Include proper paragraph breaks, formatting, and maintain the original layout as much as possible. Return only the extracted text without any additional comments."
            }
        ]
    }
    
    # Call the Gemini API with enhanced configuration
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            generation_config={
                "temperature": 0.1,  # Lower temperature for more accurate OCR
                "top_p": 0.95,       # High precision
                "top_k": 40,         # Good diversity
                "max_output_tokens": 4096  # Increased token limit for longer text
            }
        )
        return response.text
    except Exception as e:
        print(f"Error processing image with Gemini API: {e}")
        return f"Error extracting text: {str(e)}"

# Process a multi-page TIF file
def process_tif_file(tif_path, output_dir):
    """
    Process a multi-page TIF file and extract text from each page using Gemini 2.5 Pro.
    
    Args:
        tif_path: Path to the TIF file
        output_dir: Directory to save the extracted text
        
    Returns:
        Tuple of (all_text, page_texts) containing the combined text and individual page texts
    """
    print(f"Processing TIF file: {tif_path}")
    
    # Create a filename-safe base name for the output files
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    base_name = base_name.replace(" ", "_")
    
    # Open the TIF file
    img = Image.open(tif_path)
    
    # Create a directory for this TIF file's output
    tif_output_dir = os.path.join(output_dir, base_name)
    Path(tif_output_dir).mkdir(exist_ok=True, parents=True)
    
    # Find out how many pages/frames are in the file
    n_frames = getattr(img, "n_frames", 1)
    print(f"The TIF file has {n_frames} pages")
    
    # Process each page
    all_text = ""
    page_texts = []
    
    for i in range(n_frames):
        print(f"Processing page {i+1}/{n_frames}...")
        try:
            # Set the current frame to process
            img.seek(i)
            
            # Make a copy of the current frame to avoid reference issues
            current_frame = img.copy()
            
            # Extract text from the current page
            extracted_text = extract_text_from_page(current_frame)
            page_texts.append(extracted_text)
            all_text += extracted_text + "\n\n"
            
            # Save the text to a file
            page_file_path = os.path.join(tif_output_dir, f"page_{i+1:03d}.txt")
            with open(page_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            
            print(f"Saved text from page {i+1} to {page_file_path}")
            
            # Add a delay to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
    
    # Save the combined text for the entire TIF file
    combined_file_path = os.path.join(output_dir, f"{base_name}_full.txt")
    with open(combined_file_path, "w", encoding="utf-8") as f:
        f.write(all_text)
    
    # Save metadata about the extraction
    metadata = {
        "file_name": os.path.basename(tif_path),
        "pages_processed": n_frames,
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": GEMINI_MODEL
    }
    
    metadata_file_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved combined text to {combined_file_path}")
    print(f"Saved metadata to {metadata_file_path}")
    
    return all_text, page_texts

def process_entire_book(input_dir, output_dir):
    """
    Process an entire book by finding all TIF files in the input directory
    and extracting text from each chapter.
    
    Args:
        input_dir: Directory containing the TIF files
        output_dir: Directory to save the extracted text
        
    Returns:
        Combined text of the entire book
    """
    print(f"Processing all chapters in {input_dir}...")
    
    # Find all TIF files in the input directory
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return None
    
    print(f"Found {len(tif_files)} TIF files")
    
    # Process each TIF file (chapter)
    all_chapters_text = ""
    chapter_texts = {}
    
    for tif_file in tif_files:
        chapter_name = os.path.splitext(os.path.basename(tif_file))[0]
        print(f"\nProcessing chapter: {chapter_name}")
        
        chapter_text, _ = process_tif_file(tif_file, output_dir)
        chapter_texts[chapter_name] = chapter_text
        all_chapters_text += f"\n\n### {chapter_name} ###\n\n{chapter_text}"
    
    # Save the combined text for the entire book
    book_file_path = os.path.join(output_dir, "complete_book.txt")
    with open(book_file_path, "w", encoding="utf-8") as f:
        f.write(all_chapters_text)
    
    # Save metadata about the book
    book_metadata = {
        "chapters_processed": len(tif_files),
        "chapter_names": [os.path.splitext(os.path.basename(f))[0] for f in tif_files],
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": GEMINI_MODEL
    }
    
    book_metadata_path = os.path.join(output_dir, "book_metadata.json")
    with open(book_metadata_path, "w", encoding="utf-8") as f:
        json.dump(book_metadata, f, indent=2)
    
    print(f"\nEntire book processed successfully!")
    print(f"Complete book text saved to: {book_file_path}")
    print(f"Book metadata saved to: {book_metadata_path}")
    
    return all_chapters_text

def analyze_book_for_ksao(book_text, output_dir):
    """
    Analyze the entire book text to extract KSAOs using Gemini 2.5 Pro
    with maximum thinking enabled.
    
    Args:
        book_text: The full text of the book
        output_dir: Directory to save the analysis results
        
    Returns:
        The analysis result
    """
    print("Analyzing book text for KSAOs...")
    
    # Get the model
    model = client.models.get_model(GEMINI_MODEL)
    
    # Create the prompt for KSAO analysis with maximum thinking
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
    
    I NEED YOU TO DOCUMENT YOUR COMPLETE THINKING PROCESS IN EXTENSIVE DETAIL:
    1. First, carefully read through the textbook content and note key concepts related to KSAOs
    2. For each section or chapter, identify explicit and implicit competencies
    3. Categorize each KSAO, considering its nature (K, S, A, or O)
    4. Analyze relationships between KSAOs to identify hierarchies and dependencies
    5. Evaluate each KSAO's specificity, malleability, and how it's typically acquired
    6. Organize all findings into a systematic framework
    
    Show your thinking process step-by-step as you analyze the text, including your considerations,
    evaluations, reasoning, and any assumptions you make. Include your thought process about 
    how you're making categorization decisions and what patterns you notice. Make sure your 
    thinking process is exhaustive and detailed.
    
    TEXTBOOK CONTENT:
    """
    
    print(f"Text length: {len(book_text)} characters")
    
    try:
        # Send request to Gemini with enhanced settings for maximum thinking
        response = model.generate_content(
            contents=prompt + book_text,
            generation_config={
                "temperature": 0.2,       # Lower temperature for more focused analysis
                "top_p": 0.95,            # Higher precision
                "top_k": 40,              # Increased diversity
                "max_output_tokens": 8192, # Larger output for comprehensive analysis
                "thinkingBudget": 16384   # Maximum tokens for thinking (approximately 2x output)
            },
            safety_settings={
                "harassment": "block_none",
                "hate_speech": "block_none",
                "sexually_explicit": "block_none",
                "dangerous_content": "block_none"
            }
        )
        
        # Extract thinking traces and final output
        thinking_traces = ""
        final_output = ""
        
        # Check if response has candidate parts that contain the thinking traces
        if hasattr(response, 'candidates') and response.candidates:
            if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                parts = response.candidates[0].content.parts
                if len(parts) > 1:
                    # First part is thinking traces, second part is final output
                    thinking_traces = parts[0].text
                    final_output = parts[1].text
                else:
                    # If only one part, it's all combined
                    final_output = parts[0].text
        else:
            # Fallback for different response structure
            final_output = response.text
        
        # If we couldn't separate thinking and final output, use full text for both
        if not thinking_traces:
            thinking_traces = response.text
        if not final_output:
            final_output = response.text
        
        # Save the thinking traces
        thinking_file_path = os.path.join(output_dir, "ksao_thinking_process.txt")
        with open(thinking_file_path, "w", encoding="utf-8") as f:
            f.write(thinking_traces)
        
        # Save the KSAO list (final output)
        ksao_file_path = os.path.join(output_dir, "ksao_list.txt")
        with open(ksao_file_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        
        # Also save the full combined response for reference
        full_file_path = os.path.join(output_dir, "ksao_full_response.txt")
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print(f"Thinking process saved to: {thinking_file_path}")
        print(f"KSAO list saved to: {ksao_file_path}")
        print(f"Full response saved to: {full_file_path}")
        
        return response.text
        
    except Exception as e:
        print(f"Error analyzing book for KSAOs: {e}")
        
        # If the text is too long, suggest chunking
        if "content too long" in str(e).lower():
            print("The book text is too long for the model. Consider implementing a chunking strategy.")
            # Create a file with this error message
            error_file_path = os.path.join(output_dir, "analysis_error.txt")
            with open(error_file_path, "w", encoding="utf-8") as f:
                f.write(f"Error analyzing book: {e}\n\nThe book text is too long for the model. Consider implementing a chunking strategy.")
        
        return None

def main():
    parser = argparse.ArgumentParser(description="Process textbook TIF files and extract KSAOs")
    parser.add_argument("--input-dir", default="Scan", 
                      help="Directory containing TIF files (default: Scan)")
    parser.add_argument("--output-dir", default="output/full_analysis",
                      help="Directory to save output (default: output/full_analysis)")
    parser.add_argument("--ocr-only", action="store_true",
                      help="Only perform OCR without KSAO analysis")
    parser.add_argument("--analyze-only", action="store_true",
                      help="Only perform KSAO analysis on existing text file")
    parser.add_argument("--book-file", default="complete_book.txt",
                      help="Name of the book file for analysis (default: complete_book.txt)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    book_text = None
    
    # Determine workflow based on arguments
    if args.analyze_only:
        # Only perform KSAO analysis on existing text file
        book_file_path = os.path.join(args.output_dir, args.book_file)
        if os.path.exists(book_file_path):
            print(f"Loading existing book text from {book_file_path}")
            with open(book_file_path, "r", encoding="utf-8") as f:
                book_text = f.read()
            
            # Perform KSAO analysis
            analyze_book_for_ksao(book_text, args.output_dir)
        else:
            print(f"Error: Book file {book_file_path} not found!")
            print("Run the script without --analyze-only first to generate the book text.")
    
    elif args.ocr_only:
        # Only perform OCR without KSAO analysis
        print("Performing OCR only...")
        process_entire_book(args.input_dir, args.output_dir)
    
    else:
        # Perform both OCR and KSAO analysis
        print("Performing OCR followed by KSAO analysis...")
        book_text = process_entire_book(args.input_dir, args.output_dir)
        
        if book_text:
            # Perform KSAO analysis
            analyze_book_for_ksao(book_text, args.output_dir)
        else:
            print("Error: OCR process did not produce book text. KSAO analysis skipped.")
    
    print("Process completed!")

if __name__ == "__main__":
    main()