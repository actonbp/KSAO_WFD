# Migration Guide

This guide helps users adapt to the new repository structure. The project has been reorganized to focus on KSAO extraction using the Gemini API and optimize the directory structure.

## Key Changes

1. **Modern Directory Structure**:
   - `src/`: Source code organized by functionality
     - `src/ksao/`: KSAO extraction and analysis scripts
     - `src/extraction/`: Text extraction and OCR scripts
     - `src/analysis/`: Analysis utilities
     - `src/visualization/`: Visualization tools
     - `src/utils/`: Utility functions
     - `src/main.py`: Main entry point for all functionality
   - `data/`: Input data files
     - `data/gemini_text_output/`: Extracted text from textbook chapters
     - `data/Scan/`: Document scan files
     - `data/images/`: Image files for processing
   - `output/`: Generated outputs
     - `output/full_analysis/`: Results of the Gemini-based KSAO analysis
     - `output/network_visualizations/`: Network graphs of KSAO relationships
   - `docs/`: Documentation and presentations
   - `archive/`: Archived old files and approaches

2. **New Unified Entry Point**:
   - `src/main.py` provides a unified interface for all functionality
   - Commands are now provided as subcommands with options

3. **Gemini-based Analysis**:
   - Switched from using traditional NLP to Google's Gemini 2.5 Pro Preview
   - Leverages larger context windows for processing entire documents at once

## How to Adapt

### If you were running individual scripts:

**Old approach**:
```bash
python analyze_full_textbook.py
python visualize_ksao_network.py
```

**New approach**:
```bash
python src/main.py analyze
python src/main.py visualize
```

### If you were looking for output files:

**Old locations**:
- `full_analysis/textbook_ksao_analysis.txt`
- `network_visualizations/ksao_network_graph.png`

**New locations**:
- `output/full_analysis/textbook_ksao_analysis.txt`
- `output/network_visualizations/ksao_network_graph.png`

### If you were working with data:

**Old locations**:
- `gemini_text_output/Chapter_1_full.txt`

**New locations**:
- `data/gemini_text_output/Chapter_1_full.txt`

## New Features

The reorganized repository includes:

1. **Improved KSAO Extraction**: Using Gemini 2.5 Pro Preview for more accurate KSAO extraction
2. **Network Visualization**: Graph-based visualization of KSAO relationships
3. **Centralized Entry Point**: Unified command structure for all operations
4. **Proper Code Organization**: Modular structure following Python best practices

## Getting Started with the New Structure

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. Run the analysis:
   ```bash
   python src/main.py analyze
   ```

See README.md and docs/gemini_approach.md for full documentation.

## Still Need Help?

If you have questions about the new structure:

1. Check the README.md files in each directory for detailed documentation
2. See the REORGANIZATION_PLAN.md file for the full reorganization rationale
3. Review the SUMMARY_OF_CHANGES.md file for a complete list of changes