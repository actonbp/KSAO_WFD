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
  - `src/extraction/`: Text extraction and OCR scripts
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
  - `docs/ksao_framework_report.html/pdf`: Comprehensive KSAO framework report
- `docs/thinking_trace_analysis_report.html/pdf`: Analysis of AI methodological approaches
- `archive/`: Archived old files and approaches

## Current Status and Next Steps

The project has extracted and categorized KSAOs from Chapters 1-2 and Appendices of the CASAC textbook, but **this represents only a partial analysis**:

- Preliminary KSAO framework established from limited chapters
- Hierarchical relationships and developmental sequences mapped
- Methodological insights documented through thinking trace analysis
- Preliminary reports generated:
  - Framework report: `docs/ksao_framework_report.html`
  - Methodology report: `docs/thinking_trace_analysis_report.html`

### CRITICAL NEXT STEPS:

**Full processing of ALL chapters is required for a complete analysis. The current implementation with only 3 of 11 chapters is INCOMPLETE and should be considered preliminary only.**

1. **Complete OCR for all remaining chapters (3-9 and PP202)** using batch processing to overcome timeout limitations
2. **Run KSAO analysis on all remaining chapters** individually
3. **Regenerate the integrated framework** with ALL chapters included
4. **Rerun the thinking trace analysis** on the complete set of traces
5. **Update reports** with the comprehensive results

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

### Usage

**IMPORTANT: ALL chapters must be processed for a complete analysis. Do NOT skip chapters or only analyze a subset of the textbook, as this will result in an incomplete KSAO framework.**

Process each component of the workflow separately in this order:

```bash
# 1. COMPLETE OCR FOR ALL CHAPTERS
# Process each chapter, using batch processing for larger files
./process_chapter_batches.sh "Chapter 1.tif" 5
./process_chapter_batches.sh "Chapter 2.tif" 5
./process_chapter_batches.sh "Chapter 3.tif" 5
./process_chapter_batches.sh "Chapter 4.tif" 5
./process_chapter_batches.sh "Chapter 5.tif" 5
./process_chapter_batches.sh "Chapter 6.tif" 5
./process_chapter_batches.sh "Chapter 7.tif" 5
./process_chapter_batches.sh "Chapter 8.tif" 5
./process_chapter_batches.sh "Chapter 9.tif" 5
./process_chapter_batches.sh "Appendices.tif" 5
./process_chapter_batches.sh "PP202.tif" 5

# Alternative approach for smaller files or specific page ranges
python3 gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter 1.tif"
python3 gemini_ocr.py --single-chapter "Chapter 2.tif" --start-page 1 --end-page 10  # Process specific page range

# 2. RUN KSAO ANALYSIS ON EACH CHAPTER
# For each chapter, create a temp directory with just that chapter's text file
mkdir -p data/gemini_text_output/temp_Chapter_3
cp data/gemini_text_output/Chapter_3_full.txt data/gemini_text_output/temp_Chapter_3/
python3 src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output/temp_Chapter_3 --output-dir output/full_analysis --output-file Chapter_3_ksao_analysis.txt

# Repeat for ALL chapters (1-9, Appendices, PP202)
# DO NOT SKIP ANY CHAPTERS - a complete analysis is required

# 3. INTEGRATE ALL KSAO ANALYSES (after completing ALL chapters)
python3 src/ksao/integrate_ksao_analyses.py --input-dir output/full_analysis --output-dir output/full_analysis

# 4. ANALYZE ALL THINKING TRACES
python3 src/ksao/analyze_thinking_traces.py --input-dir output/full_analysis --output-dir output/full_analysis

# 5. GENERATE FINAL REPORTS
./render_reports.sh
```

A unified workflow script is also available (but not recommended for production use due to potential timeouts):

```bash
# Process OCR on document images
python gemini_ocr.py --input-dir Scan --output-dir data/gemini_text_output --single-chapter "Chapter 1.tif"

# Run KSAO analysis on a specific chapter
python src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output --output-dir output/full_analysis

# Integrate KSAO analyses across chapters
python src/ksao/integrate_ksao_analyses.py --input-dir output/full_analysis --output-dir output/full_analysis

# Analyze thinking traces across chapters
python src/ksao/analyze_thinking_traces.py --input-dir output/full_analysis --output-dir output/full_analysis
```

### Testing

To test Gemini API connectivity:
```bash
python test_gemini_api.py
```

For a simplified KSAO extraction test:
```bash
python test_simplified.py
```

## Documentation

- `CLAUDE.md`: Guide for AI assistants working with this project
- `docs/gemini_approach.md`: Detailed information about the Gemini-based approach
- `docs/ksao_analysis_report.html`: Comprehensive report of KSAO analysis results
- `docs/presentations/`: Presentations about the project

## Future Improvements

1. **Expanded Data Sources**: Include additional textbooks and certification materials
2. **Competency Validation**: Compare extracted KSAOs with expert evaluations
3. **Cross-Domain Analysis**: Extend approach to other professional domains
4. **Assessment Framework**: Develop tools to measure competency acquisition
5. **Knowledge Graph**: Build explicit relationships between competencies