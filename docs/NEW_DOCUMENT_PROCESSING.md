# Guide for Processing New Documents through the KSAO Pipeline

This guide explains how to process new documents through the KSAO analysis pipeline. Whether you're adding new textbooks, manuals, or other materials to the analysis, follow these steps to ensure consistent processing.

## Prerequisites

1. Ensure you have set up your environment:
   - Python 3.9+ installed 
   - Required dependencies: `pip install -r requirements.txt`
   - Gemini API key in a `.env` file (GEMINI_API_KEY=your_key_here)
   - ImageMagick installed (for batch processing): `brew install imagemagick`

2. Understand the document structure requirements:
   - TIFF files are the primary input format
   - PDF files need to be converted to TIFF for processing
   - Each distinct document should be in a separate TIFF file

## Step-by-Step Processing for New Documents

### 1. Prepare Your Documents

For TIFF files:
```bash
# 1. Place your TIFF files in the Scan directory
# Example: copying from another location
cp /path/to/your/document.tif /Users/acton/Documents/GitHub/KSAO_WFD/Scan/

# 2. Rename the file to follow the naming convention
# Example: for a new substance abuse manual
mv /Users/acton/Documents/GitHub/KSAO_WFD/Scan/document.tif /Users/acton/Documents/GitHub/KSAO_WFD/Scan/"Manual 1.tif"
```

For PDF files:
```bash
# Convert PDF to TIFF (using ImageMagick)
convert -density 300 document.pdf -depth 8 /Users/acton/Documents/GitHub/KSAO_WFD/Scan/"Manual 1.tif"
```

### 2. Process OCR Using Batch Processing

The recommended approach is to use batch processing, which handles the file in small batches to avoid API timeouts:

```bash
# Process in batches of 3 pages (optimal size) with output redirection
./process_chapter_batches.sh "Manual 1.tif" 3 > manual1_batch_log.txt 2>&1 &

# Monitor progress by checking the log file periodically
tail -f manual1_batch_log.txt

# Check if files are being generated
ls -la data/gemini_text_output/Manual_1/
```

For very large documents, you may want to process specific page ranges:

```bash
# Process specific page ranges
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Manual 1.tif" --start-page 1 --end-page 10
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Manual 1.tif" --start-page 11 --end-page 20
# Continue with other ranges
```

### 3. Run KSAO Analysis on the Extracted Text

After OCR processing completes, run the KSAO analysis:

```bash
# Create a temporary directory for isolated processing
mkdir -p data/gemini_text_output/temp_Manual_1
cp data/gemini_text_output/Manual_1_full.txt data/gemini_text_output/temp_Manual_1/

# Run KSAO analysis (redirect output to keep process running in background)
python3 src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output/temp_Manual_1 --output-dir output/full_analysis --output-file Manual_1_ksao_analysis.txt > manual1_analysis_log.txt 2>&1 &

# Monitor progress
tail -f manual1_analysis_log.txt
```

### 4. Integrate New Analysis with Existing Analyses

After analyzing multiple documents, integrate them:

```bash
# Integrate all analyses in the output/full_analysis directory
python3 src/ksao/integrate_ksao_analyses.py --input-dir output/full_analysis --output-dir output/full_analysis
```

### 5. Analyze Thinking Traces 

Perform meta-analysis of the thinking processes:

```bash
python3 src/ksao/analyze_thinking_traces.py --input-dir output/full_analysis --output-dir output/full_analysis
```

### 6. Generate Updated Reports

Update reports to include the new documents:

```bash
./render_reports.sh
```

## Processing Multiple Documents in Parallel

For efficiency, you can process multiple documents simultaneously:

```bash
# Start OCR on first document
./process_chapter_batches.sh "Manual 1.tif" 3 > manual1_batch_log.txt 2>&1 &

# Start OCR on second document
./process_chapter_batches.sh "Manual 2.tif" 3 > manual2_batch_log.txt 2>&1 &

# Run KSAO analysis on a third document that already completed OCR
python3 src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output/temp_Manual_3 --output-dir output/full_analysis --output-file Manual_3_ksao_analysis.txt > manual3_analysis_log.txt 2>&1 &
```

## Troubleshooting Common Issues

1. **API Timeouts During OCR**:
   - Reduce batch size to 2-3 pages
   - Process during off-peak hours
   - Ensure your API key has sufficient quota

2. **Poor OCR Quality**:
   - Check the original scan quality
   - For pages with "[LLM OCR returned no text for this page]" placeholders, try reprocessing those specific pages
   - Consider image preprocessing (adjusting contrast, etc.) before OCR

3. **KSAO Analysis Failures**:
   - Check the extracted text quality
   - Process chapters/documents individually
   - If document is very large, consider splitting it into logical sections

4. **Integration Issues**:
   - If integration fails due to input size, integrate smaller subsets of analyses first
   - Ensure all individual analyses completed successfully

## Future Improvements (Upcoming)

As noted in the README, these future improvements are planned:

1. **Consolidated Input Structure**:
   - Future document processing will use a cleaner directory structure:
     ```
     input_documents/
     ├── document_set_1/
     │   ├── raw_file_1.tif
     │   └── raw_file_2.tif
     ├── document_set_2/
     │   └── manual.pdf
     ```

2. **Enhanced Preprocessing**:
   - Improved PDF to TIFF conversion
   - Image quality enhancement for better OCR
   - Automatic handling of mixed document types

3. **Parallelized Processing**:
   - Automated multi-document processing
   - Improved monitoring and error recovery

Until these improvements are implemented, follow the current pipeline as outlined above.

## Result Verification

After processing, verify your results:

1. Check that all pages were properly extracted:
   ```bash
   ls -la data/gemini_text_output/Manual_1/
   cat data/gemini_text_output/Manual_1_full.txt | wc -l
   ```

2. Verify KSAO analysis was generated:
   ```bash
   ls -la output/full_analysis/Manual_1_ksao_analysis.txt
   ls -la output/full_analysis/Manual_1_ksao_analysis.txt_thinking_process.txt
   ```

3. Check that the new document was included in the integration:
   ```bash
   grep "Manual_1" output/full_analysis/integrated_ksao_framework.txt
   ```

4. Ensure the reports reflect the new content by opening them in a browser:
   ```bash
   open docs/ksao_framework_report.html
   ```