# Guide for AI Assistants Working with KSAO_WFD Project

## Project Overview

This repository contains a project focused on redesigning workforce development in addiction counseling by analyzing Knowledge, Skills, Abilities, and Other characteristics (KSAOs) using Gemini AI and network analysis. The project processes textbook materials for Credentialed Alcoholism and Substance Abuse Counselor (CASAC) certification to identify and categorize KSAOs, creating a more evidence-based approach to professional development.

## Repository Structure

- `src/`: Source code organized by functionality
  - `src/ksao/`: KSAO extraction and analysis scripts
  - `src/extraction/`: Text extraction and OCR scripts
  - `src/analysis/`: Analysis utilities
  - `src/visualization/`: Visualization tools
  - `src/utils/`: Utility functions
  - `src/main.py`: Main entry point for all functionality

- `data/`: Input data files
  - `data/gemini_text_output/`: Extracted text from textbook chapters
  - `data/Scan/`: Original TIFF document scans
  - `data/images/`: Image files for processing

- `output/`: Generated outputs
  - `output/full_analysis/`: Results of the Gemini-based KSAO analysis
  - `output/network_visualizations/`: Network graphs of KSAO relationships
  - `output/study_guides/`: Generated study materials
  - `output/visualizations/`: Various visualization outputs

- `docs/`: Documentation and presentations about the project
- `archive/`: Archived old files and approaches

## Main Workflow

The project follows this 5-step workflow:

1. **OCR/Text Extraction Pipeline**:
   - Input: TIFF scans in `data/Scan/` directory
   - Process: Run OCR using Gemini 2.5 multimodal capabilities via `gemini_ocr.py`
   - Output: Extracted text in `data/gemini_text_output/`

2. **KSAO Analysis Pipeline (Per Chapter)**:
   - Input: Chapter text files in `data/gemini_text_output/`
   - Process: Run KSAO analysis with thinking traces using Gemini via `src/ksao/analyze_full_textbook.py`
   - Output: Detailed KSAO analysis and thinking process files in `output/full_analysis/`

3. **KSAO Integration Pipeline**:
   - Input: Individual chapter KSAO analyses from `output/full_analysis/`
   - Process: Integrate chapter-level analyses into a comprehensive framework via `src/ksao/integrate_ksao_analyses.py`
   - Output: Integrated KSAO framework in `output/full_analysis/integrated_ksao_framework.txt`

4. **Thinking Trace Analysis Pipeline**:
   - Input: Thinking process files from `output/full_analysis/`
   - Process: Analyze thinking traces across chapters via `src/ksao/analyze_thinking_traces.py`
   - Output: Meta-analysis of thinking processes in `output/full_analysis/thinking_trace_analysis.txt`

5. **Reporting Pipeline**:
   - Input: KSAO analyses and thinking trace analysis
   - Process: Generate comprehensive report using Quarto
   - Output: HTML and PDF reports in `docs/`

## Running the Analysis

**CRITICAL REQUIREMENT: ALL chapters must be processed for a complete analysis. The current implementation with only 3 of 11 chapters is INCOMPLETE.**

The analysis workflow must be executed in sequence for ALL chapters:

```bash
# 1. COMPLETE OCR FOR ALL CHAPTERS
# Process in batches to avoid timeouts (preferred approach)
./process_chapter_batches.sh "Chapter 1.tif" 5
./process_chapter_batches.sh "Chapter 2.tif" 5
./process_chapter_batches.sh "Chapter 3.tif" 5
# ... Repeat for all chapters (4-9, Appendices, PP202)

# Alternative: Process specific page ranges manually
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter 3.tif" --start-page 1 --end-page 5
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter 3.tif" --start-page 6 --end-page 10
# ... Continue until all pages are processed

# 2. RUN KSAO ANALYSIS ON EACH CHAPTER INDIVIDUALLY
# Process each chapter by isolating its text file
mkdir -p data/gemini_text_output/temp_Chapter_3
cp data/gemini_text_output/Chapter_3_full.txt data/gemini_text_output/temp_Chapter_3/
python3 src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output/temp_Chapter_3 --output-dir output/full_analysis --output-file Chapter_3_ksao_analysis.txt
# ... Repeat for all chapters (1-9, Appendices, PP202)

# 3. INTEGRATE ALL KSAO ANALYSES (only after completing ALL chapters)
python3 src/ksao/integrate_ksao_analyses.py --input-dir output/full_analysis --output-dir output/full_analysis

# 4. ANALYZE ALL THINKING TRACES
python3 src/ksao/analyze_thinking_traces.py --input-dir output/full_analysis --output-dir output/full_analysis

# 5. GENERATE FINAL REPORTS
./render_reports.sh
```

A unified workflow script (`run_complete_workflow.py`) is also available but not recommended for production use due to potential timeouts.

**DO NOT skip chapters or create reports based on partial analysis. A comprehensive KSAO framework requires analyzing the entire textbook.**

## Important Notes for AI Assistants

1. **Gemini API Integration**:
   - The project uses Google's Gemini 2.5 Pro Preview model (gemini-2.5-pro-preview-05-06)
   - All API calls require a valid API key stored in `.env`
   - The implementation uses the thinking trace feature with a budget of 24576 tokens
   - Check `test_gemini_api.py` to verify API connectivity

2. **Workflow Structure**:
   - The project now follows a 5-step workflow as described above
   - Each step builds on the previous step's outputs
   - Individual chapters are processed separately before integration
   - Thinking traces are captured throughout and analyzed for meta-insights

3. **Handling Large Documents**:
   - OCR processes TIFF files page by page
   - Large TIFF files should be processed in smaller batches (5-10 pages at a time) using the page range parameters
   - Each chapter is analyzed individually to avoid API timeouts
   - The integration step combines analyses from multiple chapters
   - Gemini 2.5 Pro supports up to 1 million tokens of context, but API calls may time out for very large inputs

4. **Directory Structure**:
   - The repository follows a modular structure with clear separation of concerns
   - Always maintain this structure when adding new files
   - If working with code, follow existing patterns and coding style

5. **Output Interpretation**:
   - KSAO analyses produce structured output with detailed competency breakdowns
   - Each analysis includes a "thinking process" file that shows the model's reasoning
   - The integration step produces a comprehensive framework
   - The thinking trace analysis provides meta-insights about the process
   - Results are organized by Knowledge, Skills, Abilities, and Other characteristics

## Extending the Project

When extending the project to new documents, follow this approach:

1. For TIFF files:
   - Place scanned files in `data/Scan/`
   - Run the OCR process
   - Continue with KSAO analysis

2. For PDF files:
   - Convert to TIFF or use PDF-specific extraction tools
   - Process as with TIFF files

3. For developing new extraction methods:
   - Add scripts to appropriate directories (e.g., `src/extraction/`)
   - Update `src/main.py` to include new functionality
   - Document new approaches in `docs/`

## Common Issues and Solutions

- **API Timeout in OCR**: If the OCR process times out, use the `--start-page` and `--end-page` parameters to process TIFF files in smaller batches (5-10 pages at a time) via `gemini_ocr.py` or use `process_chapter_batches.sh`.
- **API Timeout in Analysis**: If the Gemini API times out during KSAO analysis, ensure you're analyzing one chapter at a time
- **Missing Dependencies**: Ensure all requirements are installed via `pip install -r requirements.txt`
- **OCR Quality**: 
    - **Updated 2024-08-01**: `gemini_ocr.py` now uses an enhanced prompt to better handle complex pages (images, layouts, tables) and includes an automated retry mechanism for pages that initially return no text. 
    - If OCR quality is still poor for certain pages (resulting in `"[LLM OCR returned no text for this page]"` placeholders even after retries), inspect the source TIFF images for those pages. Issues could be blank pages, very poor scan quality, or extremely unconventional formatting.
    - Consider adjusting image preprocessing if feasible, or documenting these pages as un-OCRable by the current method.
- **Integration Issues**: If the integration step fails due to large input size, reduce the number of chapters being integrated simultaneously

## Recent Work

The most recent work performed on the project was:
1. Restructured the workflow into a comprehensive 5-step pipeline:
   - OCR with Gemini 2.5 multimodal capabilities
   - Chapter-by-chapter KSAO extraction with thinking traces
   - Integration of chapter analyses into a unified framework
   - Meta-analysis of thinking traces across chapters
   - Report generation via Quarto
2. Created a unified `run_complete_workflow.py` script to execute the full pipeline
3. Added thinking trace capture and analysis with the maximum budget of 24576 tokens
4. Implemented advanced Gemini API configurations for optimal results
5. Added batch processing capabilities with page range parameters
6. **Enhanced OCR Robustness (August 2024)**:
   - Updated the OCR prompt in `gemini_ocr.py` (`extract_text_from_page`) to be more detailed, instructing the AI to describe images, complex layouts, and tables if direct text extraction is difficult, while still prioritizing full text extraction.
   - Implemented an automated retry mechanism (up to 2 retries with increasing delays) in `gemini_ocr.py` (`process_tif_file`) for pages where the initial OCR attempt returns no text. A placeholder (`"[LLM OCR returned no text for this page]"`) is used if all retries fail.

**IMPORTANT: The current state only includes processing for 3 chapters (1, 2, and Appendices). This is INCOMPLETE. ALL chapters (1-9 plus PP202) must be processed for a complete analysis. Any reports or frameworks generated from partial analysis should be considered preliminary only.**