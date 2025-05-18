# Data Directory

This directory contains all input data used by the KSAO Workforce Development project.

## Directory Structure

- `gemini_text_output/`: Contains extracted text from textbook chapters
  - `Chapter_1_full.txt`: Full text of Chapter 1
  - `Chapter_1_metadata.json`: Metadata for Chapter 1
  - Additional chapter files as they are processed
- `Scan/`: Contains original document scans to be processed
- `images/`: Contains image files from documents that need OCR processing
- `text_output/`: Contains extracted text from previous analysis pipeline

## Data Formats

### Gemini Text Output Format

The `gemini_text_output` directory contains files in the following format:

1. **Full Text Files**: `{Chapter_Name}_full.txt` - Contains the complete extracted text for a chapter
2. **Metadata Files**: `{Chapter_Name}_metadata.json` - Contains metadata such as source information, extraction date, etc.

### Expected Input Format

For the Gemini-based KSAO analysis, the system expects:

1. Text files containing chapter contents in the `gemini_text_output` directory
2. File naming convention: `{Chapter_Name}_full.txt`

## Adding New Data

To add new chapters or documents:

1. For already digitized content:
   - Save the text file with naming convention `{Chapter_Name}_full.txt` in the `gemini_text_output` directory
   - Create a metadata file if needed

2. For documents needing OCR:
   - Place document images/scans in the `Scan/` directory
   - Run the OCR process: `python src/main.py ocr`
   - The extracted text will be saved to `gemini_text_output`