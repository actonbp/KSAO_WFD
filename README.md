# KSAO Workforce Development - Addiction Counseling Competency Analysis

A project focused on redesigning workforce development in addiction counseling by analyzing Knowledge, Skills, Abilities, and Other characteristics (KSAOs) using Gemini AI and network analysis.

## Project Overview

This project aims to transform workforce development in addiction counseling by moving beyond traditional certification requirements to focus on actual competencies. We analyze CASAC (Credentialed Alcoholism and Substance Abuse Counselor) certification materials using Gemini AI to identify and categorize KSAOs, creating a more evidence-based approach to professional development.

## Project Goals

1. **Extract KSAOs**: Identify the core competencies required for effective addiction counseling
2. **Create Competency Maps**: Visualize relationships between different knowledge areas and skills
3. **Develop KSAO Framework**: Build a structured approach to addiction counseling competencies
4. **Enable Modernization**: Support the redesign of workforce development programs

## Current Approach

The project uses a comprehensive 5-step pipeline powered by Google's Gemini 2.5 Pro Preview model:

1. **OCR Processing**: Extract text from scanned textbook pages using Gemini's multimodal capabilities (via `gemini_ocr.py` and `process_chapter_batches.sh`).
   - *Recent Enhancements (August 2024)*: The OCR script (`gemini_ocr.py`) now employs a more detailed prompt to better handle complex page elements (images, layouts, tables) and includes an automated retry mechanism for pages that initially fail, improving overall robustness and text extraction quality.
2. **Chapter-Level KSAO Analysis**: Process each chapter separately with thinking traces enabled
3. **KSAO Integration**: Combine chapter analyses into a unified competency framework
4. **Thinking Trace Analysis**: Analyze reasoning patterns across chapters for meta-insights
5. **Comprehensive Reporting**: Generate detailed reports documenting competencies and methodologies

This approach leverages Gemini's 1 million token context window and thinking trace capabilities to create a robust, transparent analysis of professional competencies.

## Repository Structure

- `src/`: Source code organized by functionality
  - `src/ksao/`: KSAO extraction and analysis scripts
  - `src/extraction/`: Text extraction and OCR scripts (though primary OCR is `gemini_ocr.py` in root)
  - `src/analysis/`: Analysis utilities
  - `src/visualization/`: Visualization tools
  - `src/utils/`: Utility functions
  - `src/main.py`: Main entry point for all functionality
- `data/`: Input data files
  - `data/Scan/`: Original TIFF scans of textbook pages
  - `data/gemini_text_output/`: Extracted text from textbook chapters
- `output/`: Generated outputs
  - `output/full_analysis/`: Results of the Gemini-based KSAO analysis
  - `output/network_visualizations/`: Network graphs of KSAO relationships
- `docs/`: Documentation and presentations about the project
  - `docs/ksao_framework_report.html/pdf`: Comprehensive KSAO framework report (target)
  - `docs/thinking_trace_analysis_report.html/pdf`: Analysis of AI methodological approaches (target)
- `archive/`: Archived old files and approaches
  - `archive/old_scripts/`: Legacy Python scripts.
  - `archive/old_docs/`: Legacy Markdown documentation.
  - `archive/old_directories/`: Legacy data/output directories.

## Current Status and Next Steps

The project has successfully extracted and categorized KSAOs from all chapters (1-9) and Appendices of the CASAC textbook:

- **Chapter 1**: OCR complete, KSAO analysis complete
- **Chapter 2**: OCR complete, KSAO analysis complete
- **Chapter 3**: OCR complete, KSAO analysis complete
- **Chapter 4**: OCR complete, KSAO analysis complete
- **Chapter 5**: OCR complete, KSAO analysis complete
- **Chapter 6**: OCR complete, KSAO analysis complete
- **Chapter 7**: OCR complete, KSAO analysis complete
- **Chapter 8**: OCR complete, KSAO analysis complete
- **Chapter 9**: OCR complete, KSAO analysis complete
- **Appendices**: OCR complete, KSAO analysis complete

Current state represents a **comprehensive analysis**:
- Complete KSAO framework established from all textbook chapters
- Hierarchical relationships and developmental sequences mapped
- Methodological insights documented through thinking trace analysis
- Enhanced report templates created with professional styling:
  - Framework report: `docs/ksao_framework_report.qmd`
  - Methodology report: `docs/thinking_trace_analysis_report.qmd`
- Reports successfully generated in both HTML and PDF formats
- Detailed technical pipeline documentation included in reports
- Process for adding new documents clearly documented (see `docs/NEW_DOCUMENT_PROCESSING.md`)

### COMPLETED MILESTONES:

✅ OCR processing completed for all chapters (1-9) and Appendices
✅ KSAO analysis completed for all chapters individually
✅ Integrated framework generated with ALL chapters included
✅ Thinking trace analysis performed on the complete set of traces
✅ Final reports generated with comprehensive results using enhanced templates
✅ Repository organization improved with legacy scripts and files archived
✅ Pipeline documentation created for processing new documents

### POTENTIAL NEXT STEPS:

1. **Process Additional Documents**: Use the documented pipeline to analyze additional textbooks, manuals, or other materials (see `docs/NEW_DOCUMENT_PROCESSING.md`)
2. **PP202 Assessment**: Evaluate whether the PP202.tif document contains relevant material for KSAO analysis
3. **Repository Cleanup**: Continue to identify and archive any remaining legacy scripts and outdated files
4. **Future Enhancements**: Consider implementing improvements listed in the Future Improvements section

## Getting Started

### Prerequisites

- Python 3.9+
- Google API key for Gemini AI
- Virtual environment (recommended)
- Quarto (optional, for report generation)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd KSAO_WFD
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your Google API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Developer Notes

- **Favor Modification Over Creation**: Before writing a new script, please check the `src/` directories for relevant existing scripts that can be modified. Also, explore the `archive/old_scripts/` and `archive/old_docs/` directories, as they contain past approaches and utilities that might be adaptable.
- **Archiving Scripts**: Use the `./archive_deleted_files.sh` script to move outdated or superseded scripts and documents from the root or other locations into the `archive/` subdirectories. This helps keep the main project directories clean.

### Usage

**COMPLETE ANALYSIS ACHIEVED**: All chapters have been successfully processed through the entire pipeline.

The complete workflow follows this sequence:

```bash
# 1. OCR PROCESSING
# Process each chapter using batch processing
./process_chapter_batches.sh "Chapter X.tif" 3 > chapterX_batch_log.txt 2>&1 &

# IMPORTANT: When running these commands in Claude Code, they may time out in 
# the interface but continue running in the background on your system.
# Always check directory contents to verify progress before restarting any process:
# ls -la data/gemini_text_output/Chapter_X/
#
# For best results, run processes in the background with output redirection:
# ./process_chapter_batches.sh "Chapter X.tif" 3 > chapterX_batch_log.txt 2>&1 &

# Alternative approach for smaller files or specific page ranges
# python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter 1.tif"
# python3 gemini_ocr.py --single-chapter "Chapter 2.tif" --start-page 1 --end-page 10  # Process specific page range

# 2. KSAO ANALYSIS ON EACH CHAPTER
# For each chapter, create a temp directory with just that chapter's text file
mkdir -p data/gemini_text_output/temp_Chapter_X
cp data/gemini_text_output/Chapter_X_full.txt data/gemini_text_output/temp_Chapter_X/
python3 src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output/temp_Chapter_X --output-dir output/full_analysis --output-file Chapter_X_ksao_analysis.txt

# 3. INTEGRATION OF ALL KSAO ANALYSES
python3 src/ksao/integrate_ksao_analyses.py --input-dir output/full_analysis --output-dir output/full_analysis

# 4. THINKING TRACE ANALYSIS
python3 src/ksao/analyze_thinking_traces.py --input-dir output/full_analysis --output-dir output/full_analysis

# 5. REPORT GENERATION
./render_reports.sh
```

All steps have been successfully completed for Chapters 1-9 and Appendices. The final reports are available in both HTML and PDF formats in the `docs/` directory.

### Testing

To test Gemini API connectivity (ensure `test_gemini_api.py` is in `archive/old_scripts` or run it from there):
```bash
python archive/old_scripts/test_gemini_api.py
```

For a simplified KSAO extraction test (ensure `test_simplified.py` is in `archive/old_scripts` or run it from there):
```bash
python archive/old_scripts/test_simplified.py
```

## Documentation

- `CLAUDE.md`: Guide for AI assistants working with this project.
- `README_NEXT_STEPS.md`: Detailed breakdown of immediate tasks to complete the current analysis.
- `archive/old_docs/`: Contains older READMEs and planning documents.
- `docs/`: Contains project reports and presentations (target location for Quarto outputs).

## Future Improvements

1. **Expanded Data Sources**: Include additional textbooks and certification materials.
2. **Competency Validation**: Compare extracted KSAOs with expert evaluations.
3. **Cross-Domain Analysis**: Extend approach to other professional domains.
4. **Assessment Framework**: Develop tools to measure competency acquisition.
5. **Knowledge Graph**: Build explicit relationships between competencies.
6. **Scalable & Versatile Document Processing Pipeline & Input Structure**:
   *   **Input Format Handling**: Develop a highly robust and flexible input system capable of handling diverse document formats (e.g., PDF, TIFF, other image types). This includes creating a standardized pre-processing stage, for instance, converting PDF pages into a consistent image format suitable for the Gemini OCR pipeline. Any intermediate files generated (like page images from PDFs) should be managed by scripts and stored in a structured way within the `output/` directory (e.g., `output/intermediate_page_images/document_set_name/`).
   *   **Consolidated Input Directory**: Transition to a cleaner input file organization. For example, create a primary `input_documents/` directory. Within this, each distinct source document (e.g., a textbook, a manual) would have its own subdirectory (e.g., `input_documents/casac_textbook_main/`, `input_documents/supplementary_guide_v1/`). The original raw files (TIFFs, PDFs) for that document set would reside directly in its subdirectory.
   *   **Scalable Workflow**: Design the overall workflow and supporting scripts to easily target these individual document set subdirectories within `input_documents/`. This approach supports the goal of efficiently processing a large and diverse corpus of documents in a replicable manner, keeping raw inputs clearly separated and organized.
7. **Enhanced OCR Failure Handling & Prompt Refinement**:
   * Continuously analyze OCR failures to identify patterns (e.g., specific layouts, image types, scan quality issues) that lead to placeholder text (`\"[LLM OCR returned no text for this page]\"`).
   * Iteratively refine the OCR prompt in `gemini_ocr.py` to specifically address these failure patterns, providing more detailed instructions to the LLM on how to handle problematic pages and extract all discernible text, even from complex or partially obscured content.
8. **Performance Optimization & Parallelization**:
   * Investigate and implement strategies to accelerate the overall workflow, particularly the OCR and KSAO analysis phases.
   * Explore parallel processing of pages or chapters where feasible, leveraging multi-core CPUs (e.g., using Python's `multiprocessing` or `asyncio` for concurrent API calls).
   * Research potential for GPU acceleration if any parts of the local processing (e.g., image pre-processing if implemented) could benefit, though primary bottlenecks are likely API-related.
   * Consider architectural changes, such as a more distributed or queue-based system, if scaling to a very large number of documents, potentially involving multiple worker agents or asynchronous task management to handle many API calls efficiently.