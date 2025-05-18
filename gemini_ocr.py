import os
import base64
import argparse
from pathlib import Path
from PIL import Image
import io
from google import genai
from dotenv import load_dotenv
import time
import json
import glob

# Load environment variables from .env file (contains GEMINI_API_KEY)
load_dotenv()

# Get the API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the Gemini API client
client = genai.Client(api_key=api_key)

# Current model version
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"  # Updated to use Gemini 2.5 Pro

# Create output directory if it doesn't exist
def create_output_directory(output_dir="data/gemini_text_output"):
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
                "text": "Your primary goal is to extract ALL text from this image with the highest possible accuracy and completeness. Maintain proper paragraph breaks, formatting, and the original layout as much as possible.\\nIf the image contains significant visual elements (e.g., photographs, diagrams, charts, maps) that are not primarily text, provide a detailed description of the image content and its relevance if apparent, like '[Image: Detailed description including type, subject, and any text or labels visible within the image itself.]'. Extract all text associated with the image, such as captions or labels.\\nIf the page has a complex layout (e.g., multiple columns, sidebars, text boxes, footnotes), first describe the page structure (e.g., '[Layout: Two columns with a header and a footer. A sidebar is on the right containing X.]') and then meticulously extract the text from each section in a logical reading order.\\nIf tables are present, attempt to extract their content preserving row and column structure as accurately as possible. If direct structured extraction is too complex, describe the table's structure, like '[Table: X columns, Y rows. Column headers are: A, B, C. Briefly describe content type.]' and then extract the cell data as best as possible.\\nEnsure all text, including small print, captions, headers, footers, and any text embedded within graphical elements, is extracted.\\nReturn ONLY the extracted text and any bracketed descriptions of images, layouts, or tables. Do not summarize, interpret, or add any information not explicitly present in the image. Do not engage in any conversation or provide commentary outside of the requested bracketed descriptions."
            }
        ]
    }
    
    # Call the Gemini API with enhanced configuration
    try:
        # The API changed - use the correct method for current Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )
        # Ensure a string is always returned, even if response.text is None
        return response.text if response.text is not None else "[LLM OCR returned no text for this page]"
    except Exception as e:
        print(f"Error processing image with Gemini API: {e}")
        return f"Error extracting text: {str(e)}"

# Process a multi-page TIF file
def process_tif_file(tif_path, output_dir="data/gemini_text_output", start_page=None, end_page=None):
    """
    Process a multi-page TIF file and extract text from each page using Gemini 2.5 Pro.
    
    Args:
        tif_path: Path to the TIF file
        output_dir: Directory to save the extracted text
        start_page: Page number to start processing from (1-based index)
        end_page: Page number to end processing at (inclusive, 1-based index)
        
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
    
    # Adjust start and end page to 0-based indexing
    start_idx = start_page - 1 if start_page is not None else 0
    end_idx = end_page - 1 if end_page is not None else n_frames - 1
    
    # Validate page range
    start_idx = max(0, min(start_idx, n_frames - 1))
    end_idx = max(start_idx, min(end_idx, n_frames - 1))
    
    print(f"Processing pages {start_idx+1} to {end_idx+1} (out of {n_frames} total pages)")
    
    # Define the specific placeholder for initial OCR failure
    OCR_FAILURE_PLACEHOLDER = "[LLM OCR returned no text for this page]"
    MAX_RETRIES = 2 # Try initial + 2 retries

    for i in range(start_idx, end_idx + 1):
        print(f"Processing page {i+1}/{n_frames}...")
        page_successfully_processed = False
        extracted_text = "" # Ensure extracted_text is initialized

        for attempt in range(MAX_RETRIES + 1): # Initial attempt (0) + MAX_RETRIES
            try:
                # Set the current frame to process
                img.seek(i)
                
                # Make a copy of the current frame to avoid reference issues
                current_frame = img.copy()
                
                if attempt > 0:
                    print(f"Retrying page {i+1}/{n_frames}, attempt {attempt}...")
                    time.sleep(5 * attempt) # Increase delay for subsequent retries

                # Extract text from the current page
                extracted_text = extract_text_from_page(current_frame)
                
                if extracted_text != OCR_FAILURE_PLACEHOLDER:
                    page_successfully_processed = True
                    break # Exit retry loop if successful or if a different error string is returned
                elif attempt < MAX_RETRIES:
                    print(f"Page {i+1} OCR failed (no text returned), will retry...")
                else: # Last attempt and still the placeholder
                    print(f"Page {i+1} OCR failed after {MAX_RETRIES} retries (no text returned). Using placeholder.")
            
            except Exception as e:
                # This catches errors in img.seek, img.copy, or other unexpected issues within the try block
                print(f"Error processing page {i+1} (attempt {attempt}): {e}")
                extracted_text = f"[Error processing page {i+1}: {str(e)}]" # Ensure extracted_text is a string
                break # Exit retry loop on other errors

        # Append the result (successful or final placeholder/error)
        page_texts.append(extracted_text)
        all_text += extracted_text + "\n\n"
        
        # Save the text to a file
        page_file_path = os.path.join(tif_output_dir, f"page_{i+1:03d}.txt")
        with open(page_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text) # extracted_text is guaranteed to be a string here
        
        if page_successfully_processed:
            print(f"Saved text from page {i+1} to {page_file_path}")
        else:
            print(f"Saved placeholder/error for page {i+1} to {page_file_path}")
        
        # Add a delay to avoid hitting rate limits (applies after all attempts for a page)
        time.sleep(1)
            
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

def process_entire_book(input_dir="Scan", output_dir="data/gemini_text_output"):
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

def main():
    """Main function to run the OCR process"""
    parser = argparse.ArgumentParser(description="Extract text from TIFF images using Gemini 2.5 Pro")
    parser.add_argument("--input-dir", default="Scan", help="Directory containing TIF files")
    parser.add_argument("--output-dir", default="data/gemini_text_output", 
                        help="Directory to save extracted text")
    parser.add_argument("--single-chapter", default=None, 
                        help="Process only a specific chapter file (e.g., 'Chapter 1.tif')")
    parser.add_argument("--process-all", action="store_true", 
                        help="Process all chapters and combine into a complete book")
    parser.add_argument("--start-page", type=int, default=None,
                        help="Start processing from this page number (1-based index)")
    parser.add_argument("--end-page", type=int, default=None,
                        help="End processing at this page number (inclusive, 1-based index)")
    
    args = parser.parse_args()
    
    # Create the output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Process based on arguments
    if args.process_all:
        process_entire_book(args.input_dir, output_dir)
    elif args.single_chapter:
        tif_path = os.path.join(args.input_dir, args.single_chapter)
        if os.path.exists(tif_path):
            all_text, page_texts = process_tif_file(
                tif_path, 
                output_dir, 
                start_page=args.start_page, 
                end_page=args.end_page
            )
            processed_pages = len(page_texts)
            if args.start_page or args.end_page:
                print(f"Processed {tif_path} - extracted text from {processed_pages} pages (pages {args.start_page or 1} to {args.end_page or 'end'})")
            else:
                print(f"Processed {tif_path} - extracted text from {processed_pages} pages")
        else:
            print(f"Error: File {tif_path} does not exist")
    else:
        # Default: process Chapter 1.tif
        tif_path = os.path.join(args.input_dir, "Chapter 1.tif")
        if os.path.exists(tif_path):
            all_text, page_texts = process_tif_file(
                tif_path, 
                output_dir, 
                start_page=args.start_page, 
                end_page=args.end_page
            )
            processed_pages = len(page_texts)
            if args.start_page or args.end_page:
                print(f"Processed {tif_path} - extracted text from {processed_pages} pages (pages {args.start_page or 1} to {args.end_page or 'end'})")
            else:
                print(f"Processed {tif_path} - extracted text from {processed_pages} pages")
        else:
            print(f"Error: File {tif_path} does not exist")
            print("Use --input-dir to specify the correct directory or --single-chapter to specify a file")

if __name__ == "__main__":
    main() 