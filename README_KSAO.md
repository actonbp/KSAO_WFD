# KSAO Extraction Tool

This tool processes textbook TIF files and extracts Knowledge, Skills, Abilities, and Other Characteristics (KSAOs) required for substance use disorder (SUD) counselors. It uses Google's Gemini 2.5 Pro API for both OCR text extraction and KSAO analysis.

## Features

- Extracts text from multi-page TIF files using Gemini 2.5 Pro OCR capabilities
- Processes an entire book across multiple chapter files
- Analyzes the extracted text to identify KSAOs with detailed thinking process
- Saves both raw text output and KSAO analysis results
- Flexible workflow options (OCR-only, analysis-only, or combined)

## Requirements

- Python 3.6+
- Google Gemini API key (stored in `.env` file)
- Required Python packages:
  - google-generativeai
  - python-dotenv
  - pillow (PIL)
  - pathlib

## Setup

1. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The tool can be run in three modes:

### 1. Full Processing (OCR + KSAO Analysis)

```bash
./process_textbook.py --input-dir Scan --output-dir output/full_analysis
```

This will:
- Process all TIF files in the `Scan` directory
- Extract text from each chapter
- Combine all chapters into a single text file
- Analyze the combined text to extract KSAOs
- Save both the raw text and analysis results to the output directory

### 2. OCR Only

```bash
./process_textbook.py --ocr-only --input-dir Scan --output-dir output/full_analysis
```

This will only perform the OCR text extraction without KSAO analysis.

### 3. KSAO Analysis Only

```bash
./process_textbook.py --analyze-only --output-dir output/full_analysis --book-file complete_book.txt
```

This assumes you already have text extracted and will only perform the KSAO analysis on an existing text file.

## Output Files

The tool generates the following outputs:

### OCR Phase
- `Chapter_X/page_NNN.txt`: Individual page text for each chapter
- `Chapter_X_full.txt`: Combined text for each chapter
- `Chapter_X_metadata.json`: Metadata about the processing
- `complete_book.txt`: Combined text from all chapters
- `book_metadata.json`: Metadata about the entire book

### KSAO Analysis Phase
- `ksao_thinking_process.txt`: Detailed thinking process used for analysis
- `ksao_list.txt`: Clean list of extracted KSAOs
- `ksao_full_response.txt`: Complete raw response from the model

## Command Line Arguments

- `--input-dir`: Directory containing TIF files (default: "Scan")
- `--output-dir`: Directory to save output (default: "output/full_analysis")
- `--ocr-only`: Only perform OCR without KSAO analysis
- `--analyze-only`: Only perform KSAO analysis on existing text file 
- `--book-file`: Name of the book file for analysis (default: "complete_book.txt")

## Examples

### Process a single chapter
```bash
./process_textbook.py --input-dir Scan --output-dir output/chapter1_analysis --ocr-only
```

### Analyze existing text
```bash
./process_textbook.py --analyze-only --output-dir output/ksao_results --book-file complete_book.txt
```

## Notes

- The tool will process TIF files in alphabetical order
- For large textbooks, consider using the `--ocr-only` mode first, then analyzing chapters separately
- The Gemini 2.5 Pro model has a maximum token limit; very large books may need to be processed in chunks

## Troubleshooting

If you encounter errors like "content too long", the tool will suggest implementing a chunking strategy. You can:

1. Process each chapter separately using the `--ocr-only` option
2. Analyze chapters individually with `--analyze-only` option
3. Combine the results manually

For API rate limiting issues, the tool automatically adds small delays between requests.