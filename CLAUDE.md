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

**COMPLETE ANALYSIS ACHIEVED: All chapters (1-9) and Appendices have been successfully processed through the entire pipeline.**

The analysis workflow follows this sequence:

```bash
# 1. OCR PROCESSING
# Process each chapter using batch processing (batch size of 3 is optimal)
./process_chapter_batches.sh "Chapter X.tif" 3 > chapterX_batch_log.txt 2>&1 &

# Alternative: Process specific page ranges manually
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter X.tif" --start-page 1 --end-page 5
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter X.tif" --start-page 6 --end-page 10
# ... Continue until all pages are processed

# 2. KSAO ANALYSIS ON EACH CHAPTER INDIVIDUALLY
# Process each chapter by isolating its text file
mkdir -p data/gemini_text_output/temp_Chapter_X
cp data/gemini_text_output/Chapter_X_full.txt data/gemini_text_output/temp_Chapter_X/
python3 src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output/temp_Chapter_X --output-dir output/full_analysis --output-file Chapter_X_ksao_analysis.txt

# 3. INTEGRATION OF ALL KSAO ANALYSES
python3 src/ksao/integrate_ksao_analyses.py --input-dir output/full_analysis --output-dir output/full_analysis

# 4. THINKING TRACE ANALYSIS
python3 src/ksao/analyze_thinking_traces.py --input-dir output/full_analysis --output-dir output/full_analysis

# 5. GENERATE FINAL REPORTS
./render_reports.sh
```

A unified workflow script (`run_complete_workflow.py`) is also available but not recommended for production use due to potential timeouts.

**All chapters have been successfully processed and all reports have been generated. The analysis is now COMPLETE.**

## Important Notes for AI Assistants

1. **Core Principle - Favor Modification**: Before creating new scripts, always check `src/` and `archive/old_scripts/` for existing code that can be adapted. The goal is a lean codebase.

2. **Gemini API Integration**:
   - The project uses Google's Gemini 2.5 Pro Preview model (`gemini-2.5-pro-preview-05-06`)
   - All API calls require a valid API key stored in `.env`
   - The implementation uses the thinking trace feature with a budget of 24576 tokens
   - Check `test_gemini_api.py` to verify API connectivity

3. **Workflow Structure**:
   - The project now follows a 5-step workflow as described above
   - Each step builds on the previous step's outputs
   - Individual chapters are processed separately before integration
   - Thinking traces are captured throughout and analyzed for meta-insights

4. **Directory Structure & Inputs**:
   - The repository follows a modular structure. Maintain this when adding files.
   - **Current primary input for TIFF scans is `data/Scan/`.** The `data/images/` directory has been archived.
   - A future goal (see `README.md` Future Improvements) is a more consolidated input structure (e.g., `input_documents/<document_set_name>/raw_files`).
   - Key output directories: `data/gemini_text_output/` (for OCR text), `output/full_analysis/` (for KSAO analyses).
   - Use `./archive_deleted_files.sh` to move old scripts/docs to `archive/` subdirectories.

5. **Output Interpretation**:
   - KSAO analyses produce structured output with detailed competency breakdowns
   - Each analysis includes a "thinking process" file that shows the model's reasoning
   - The integration step produces a comprehensive framework
   - The thinking trace analysis provides meta-insights about the process
   - Results are organized by Knowledge, Skills, Abilities, and Other characteristics

## Extending the Project

The project pipeline is fully functional and ready for processing additional documents. A detailed guide for processing new documents is available in `docs/NEW_DOCUMENT_PROCESSING.md`.

When extending the project to new documents, follow this approach:

1. For TIFF files:
   - Place scanned files in `data/Scan/`
   - Run the OCR process using `process_chapter_batches.sh`
   - Continue with KSAO analysis using `src/ksao/analyze_full_textbook.py`
   - Integrate with existing analyses using `src/ksao/integrate_ksao_analyses.py`
   - Update reports using `render_reports.sh`

2. For PDF files:
   - Convert to TIFF using ImageMagick: `convert -density 300 document.pdf -depth 8 Scan/Document.tif`
   - Process the resulting TIFF file as above

3. For developing new extraction methods:
   - Add scripts to appropriate directories (e.g., `src/extraction/`)
   - Update `src/main.py` to include new functionality
   - Document new approaches in `docs/`

The entire pipeline has been tested and optimized for efficient processing of large document sets. Multiple documents can be processed simultaneously for greater efficiency.

## Common Issues and Solutions

- **Claude Code Timeouts**: When running long processes like OCR or KSAO analysis in Claude Code, the commands may time out in the Claude Code interface but continue running in the background on your system. Before restarting a process:
    - Always check if files are being generated in the appropriate directories (`data/gemini_text_output/Chapter_X/` for OCR)
    - Use `ls` commands to monitor progress (e.g., `ls -la data/gemini_text_output/Chapter_4/`)
    - Allow the background process to complete before starting new operations on the same files
    - For maximum reliability, run processes in the background with output redirection (e.g., `command > logfile.txt 2>&1 &`)
    - These processes can take hours to complete for large chapters, so be patient

- **API Timeout in OCR**: If the OCR process times out, use the `--start-page` and `--end-page` parameters to process TIFF files in smaller batches (5-10 pages at a time) via `gemini_ocr.py` or use `process_chapter_batches.sh` with a batch size of 3.
- **API Timeout in Analysis**: If the Gemini API times out during KSAO analysis, ensure you're analyzing one chapter at a time
- **Missing Dependencies**: Ensure all requirements are installed via `pip install -r requirements.txt`
- **Mac M2 Studio Optimization**: This project has been tested and optimized for the Mac M2 Studio architecture. For best performance:
    - Ensure Python 3.9+ is installed via Homebrew
    - Install MacTeX for PDF generation with Quarto
    - Keep Python dependencies updated
- **OCR Quality**: 
    - **Updated 2024-08-01**: `gemini_ocr.py` now uses an enhanced prompt to better handle complex pages (images, layouts, tables) and includes an automated retry mechanism for pages that initially return no text. 
    - If OCR quality is still poor for certain pages (resulting in `\"[LLM OCR returned no text for this page]\"` placeholders even after retries), inspect the source TIFF images for those pages. Issues could be blank pages, very poor scan quality, or extremely unconventional formatting.
    - Consider adjusting image preprocessing if feasible, or documenting these pages as un-OCRable by the current method.
    - **Ongoing Development**: We are continuously looking for patterns in OCR failures. Future work includes further iterative refinement of the OCR prompt in `gemini_ocr.py` to provide more targeted instructions to the LLM for handling common failure cases and maximizing text extraction from challenging pages.
- **Integration Issues**: If the integration step fails due to large input size, reduce the number of chapters being integrated simultaneously

## Recent Work

The most recent work performed on the project was:
1. **Completed full processing of ALL chapters (1-9) and Appendices**
2. Generated the integrated KSAO framework incorporating all chapters
3. Completed the thinking trace analysis across all chapters
4. Fixed PDF rendering issues in the report templates by adjusting LaTeX package options
5. Generated final reports in both HTML and PDF formats using enhanced templates
6. Added detailed technical pipeline documentation to the reports
7. Updated README.md and CLAUDE.md to reflect the completed project status
8. Created enhanced report templates with improved styling for both HTML and PDF outputs
9. Added multiple optimizations for the Mac M2 Studio architecture
10. Implemented a robust parallel workflow strategy using background processes and output redirection
11. Restructured the workflow into a comprehensive 5-step pipeline:
    - OCR with Gemini 2.5 multimodal capabilities
    - Chapter-by-chapter KSAO extraction with thinking traces
    - Integration of chapter analyses into a unified framework
    - Meta-analysis of thinking traces across chapters
    - Report generation via Quarto
12. Created a unified `run_complete_workflow.py` script to execute the full pipeline
13. Added thinking trace capture and analysis with the maximum budget of 24576 tokens
14. Implemented advanced Gemini API configurations for optimal results
15. Added batch processing capabilities with page range parameters
16. **Enhanced OCR Robustness (August 2024)**:
    - Updated the OCR prompt in `gemini_ocr.py` (`extract_text_from_page`) to be more detailed, instructing the AI to describe images, complex layouts, and tables if direct text extraction is difficult, while still prioritizing full text extraction.
    - Implemented an automated retry mechanism (up to 2 retries with increasing delays) in `gemini_ocr.py` (`process_tif_file`) for pages where the initial OCR attempt returns no text. A placeholder (`"[LLM OCR returned no text for this page]"`) is used if all retries fail.

**IMPORTANT: The project has been completed with all chapters (1-9) and Appendices fully processed through the entire pipeline. The only remaining optional task is evaluating PP202.tif for relevance, which may be included based on assessment.**