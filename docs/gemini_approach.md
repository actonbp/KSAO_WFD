# Gemini-based KSAO Extraction Approach

This document explains the current approach to extracting KSAOs (Knowledge, Skills, Abilities, and Other characteristics) from substance use disorder (SUD) counselor educational materials using Google's Gemini AI.

## Overview

The project has shifted from using traditional NLP techniques to using Google's Gemini 2.5 Pro Preview model, which offers several advantages:

1. **Large Context Window**: Ability to process entire textbook chapters at once
2. **Comprehensive Understanding**: Better semantic understanding of complex concepts
3. **Hierarchical Classification**: Ability to identify relationships between competencies
4. **Simplified Pipeline**: Reduces the need for complex chunking and preprocessing

## Current Workflow

The current workflow consists of the following steps:

1. **Text Preparation**: Extract textbook content using OCR or direct conversion
2. **Full Textbook Analysis**: Process the entire textbook using Gemini 2.5 Pro Preview
3. **KSAO Extraction**: Automatically identify and categorize KSAOs
4. **Network Visualization**: Create graphical representations of KSAO relationships

## Script Descriptions

### analyze_full_textbook.py

This script processes the entire textbook content using Gemini AI to extract KSAOs:

- **Input**: Textbook chapters stored as text files in the `gemini_text_output` directory
- **Process**: Combines all chapters and sends them to Gemini 2.5 Pro Preview with a specialized prompt
- **Output**: Detailed analysis of KSAOs in `full_analysis/textbook_ksao_analysis.txt`

Key features:
- Uses Gemini's large context window to process the entire textbook at once
- Employs a carefully designed prompt to focus on KSAO extraction
- Classifies competencies by type, stability/malleability, and explicit/tacit orientation

### visualize_ksao_network.py

This script creates network visualizations of the KSAO relationships:

- **Input**: KSAO analysis from `analyze_full_textbook.py`
- **Process**: Extracts structured relationship data and creates network graphs
- **Output**: Various network visualizations in the `network_visualizations` directory

Key features:
- Creates standard network graphs showing all KSAOs and their relationships
- Generates hierarchical layouts showing prerequisite/foundation relationships
- Creates classification-based subgraphs (Knowledge, Skills, Abilities, Other)

### run_ksao_analysis.py

This orchestration script runs the entire KSAO analysis process:

- **Input**: Command-line arguments specifying input/output directories
- **Process**: Sequentially runs the analysis and visualization scripts
- **Output**: Complete analysis and visualization files

Key features:
- Creates necessary output directories
- Checks for required dependencies
- Provides a simple interface for running the complete analysis

## Key Parameters and Settings

The Gemini API is configured with specific parameters to optimize KSAO extraction:

```python
model = genai_client.models.get_model("gemini-2.5-pro-preview-05-06")
response = model.generate_content(
    contents=prompt + text,
    generation_config={
        "temperature": 0.2,  # Lower temperature for more focused analysis
        "top_p": 0.95,       # Higher precision
        "top_k": 40,         # Increased diversity
        "max_output_tokens": 8192  # Larger output for comprehensive analysis
    },
    safety_settings={
        "harassment": "block_none",
        "hate_speech": "block_none",
        "sexually_explicit": "block_none",
        "dangerous_content": "block_none"
    }
)
```

## Using the Scripts

### Prerequisites

1. A Google API key with access to Gemini 2.5 Pro Preview
2. Python 3.9+ with required dependencies installed
3. Textbook chapter text files in the `gemini_text_output` directory

### Running the Analysis

The simplest way to run the complete analysis is:

```bash
python run_ksao_analysis.py
```

This will:
1. Process the textbook with Gemini AI
2. Create network visualizations
3. Save all results to the appropriate directories

### Customization Options

The scripts support various command-line arguments:

```bash
python run_ksao_analysis.py --input-dir custom_input --output-dir custom_output --no-visualize
```

- `--input-dir`: Specify a different input directory (default: `gemini_text_output`)
- `--output-dir`: Specify a different output directory (default: `full_analysis`)
- `--no-visualize`: Skip the visualization step

## Future Enhancements

Planned improvements to the Gemini-based approach:

1. **Multi-Source Integration**: Incorporate multiple textbooks and training materials
2. **Expert Validation**: Compare AI-extracted KSAOs with expert-identified competencies
3. **Task-Specific Refinement**: Fine-tune the analysis for specific counseling contexts
4. **Interactive Dashboards**: Create interactive tools for exploring the KSAO network
5. **Curriculum Mapping**: Align identified KSAOs with existing training curricula 