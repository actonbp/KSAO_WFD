import os
import base64
from pathlib import Path
from PIL import Image
import io
from google import genai
from dotenv import load_dotenv
import time
import json

# Load environment variables from .env file (contains GEMINI_API_KEY)
load_dotenv()

# Get the API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the Gemini API client
client = genai.Client(api_key=api_key)

# Create output directory if it doesn't exist
def create_output_directory(output_dir="gemini_text_output"):
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir

# Convert an image to base64 for API consumption
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Process a single page and extract text using Gemini API
def extract_text_from_page(img):
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
                "text": "Extract all text from this image. Return only the extracted text without any additional comments or formatting."
            }
        ]
    }
    
    # Call the Gemini API
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro-latest",  # Using a model that can process images
            contents=contents
        )
        return response.text
    except Exception as e:
        print(f"Error processing image with Gemini API: {e}")
        return f"Error extracting text: {str(e)}"

# Process a multi-page TIF file
def process_tif_file(tif_path, output_dir="gemini_text_output"):
    print(f"Processing TIF file: {tif_path}")
    
    # Create a filename-safe base name for the output files
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    base_name = base_name.replace(" ", "_")
    
    # Open the TIF file
    img = Image.open(tif_path)
    
    # Create a directory for this TIF file's output
    tif_output_dir = os.path.join(output_dir, base_name)
    Path(tif_output_dir).mkdir(exist_ok=True)
    
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
            with open(page_file_path, "w") as f:
                f.write(extracted_text)
            
            print(f"Saved text from page {i+1} to {page_file_path}")
            
            # Add a delay to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
    
    # Save the combined text for the entire TIF file
    combined_file_path = os.path.join(output_dir, f"{base_name}_full.txt")
    with open(combined_file_path, "w") as f:
        f.write(all_text)
    
    # Save metadata about the extraction
    metadata = {
        "file_name": os.path.basename(tif_path),
        "pages_processed": n_frames,
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": "gemini-1.5-pro-latest"
    }
    
    metadata_file_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved combined text to {combined_file_path}")
    print(f"Saved metadata to {metadata_file_path}")
    
    return all_text, page_texts

def main():
    # Create the output directory
    output_dir = create_output_directory()
    
    # For now, just process Chapter 1.tif
    tif_path = "Scan/Chapter 1.tif"
    
    if os.path.exists(tif_path):
        all_text, page_texts = process_tif_file(tif_path, output_dir)
        print(f"Processed {tif_path} - extracted text from {len(page_texts)} pages")
    else:
        print(f"Error: File {tif_path} does not exist")

if __name__ == "__main__":
    main() 