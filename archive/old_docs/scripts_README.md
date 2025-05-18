# KSAO Analysis Scripts

This directory contains the scripts used for extracting KSAOs (Knowledge, Skills, Abilities, and Other characteristics) from substance use disorder (SUD) counselor textbooks using Google's Gemini 2.5 Pro Preview model.

## Script Overview

### src/ksao/analyze_full_textbook.py

This script processes the entire textbook content using Gemini AI to extract KSAOs:

```bash
python src/ksao/analyze_full_textbook.py --input-dir data/gemini_text_output --output-dir output/full_analysis
```

**Parameters:**
- `--input-dir`: Directory containing the textbook chapter text files (default: `data/gemini_text_output`)
- `--output-dir`: Directory for the analysis output (default: `output/full_analysis`)
- `--output-file`: Name of the output file (default: `textbook_ksao_analysis.txt`)

### src/ksao/visualize_ksao_network.py

This script creates network visualizations of the KSAOs and their relationships:

```bash
python src/ksao/visualize_ksao_network.py --analysis-file output/full_analysis/textbook_ksao_analysis.txt --output-dir output/network_visualizations
```

**Parameters:**
- `--analysis-file`: File containing the textbook analysis (default: `output/full_analysis/textbook_ksao_analysis.txt`)
- `--output-dir`: Directory for visualization output (default: `output/network_visualizations`)

### src/ksao/run_ksao_analysis.py

This is a script that runs the complete KSAO analysis process:

```bash
python src/ksao/run_ksao_analysis.py --input-dir data/gemini_text_output --output-dir output/full_analysis
```

**Parameters:**
- `--input-dir`: Directory containing the textbook chapter text files (default: `data/gemini_text_output`)
- `--output-dir`: Directory for the analysis output (default: `output/full_analysis`)
- `--no-visualize`: Skip the visualization step (flag, no value needed)

### src/main.py

This is the main entry point for the entire project, providing commands for all functionality:

```bash
python src/main.py analyze   # Run the complete analysis pipeline
python src/main.py ocr       # Run OCR on document images
python src/main.py visualize # Create visualizations from existing analysis
```

**Parameters for 'analyze' command:**
- `--input-dir`: Directory containing the textbook chapter text files (default: `data/gemini_text_output`)
- `--output-dir`: Directory for the analysis output (default: `output/full_analysis`)
- `--no-visualize`: Skip the visualization step (flag, no value needed)

**Parameters for 'ocr' command:**
- `--input-dir`: Directory containing document images (default: `data/Scan`)
- `--output-dir`: Directory for OCR output (default: `data/gemini_text_output`)

**Parameters for 'visualize' command:**
- `--analysis-file`: File containing the textbook analysis (default: `output/full_analysis/textbook_ksao_analysis.txt`)
- `--output-dir`: Directory for visualization output (default: `output/network_visualizations`)

## Setup Requirements

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with your Google API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Ensure you have textbook chapter text files in the input directory (`data/gemini_text_output` by default)

## Usage Example

To run the complete analysis process:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key in .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the complete analysis
python src/main.py analyze
```

This will process the textbook chapters, perform KSAO analysis using Gemini 2.5 Pro Preview, and create network visualizations of the results.

## Expected Outputs

- `output/full_analysis/textbook_ksao_analysis.txt`: Detailed KSAO analysis from Gemini
- `output/full_analysis/textbook_ksao_analysis_combined_text.txt`: The combined textbook text sent to Gemini
- `output/network_visualizations/ksao_network_data.json`: JSON representation of the KSAO network
- `output/network_visualizations/ksao_network_graph.png`: Visual network graph of all KSAOs
- `output/network_visualizations/ksao_hierarchical_graph.png`: Hierarchical representation of KSAOs
- `output/network_visualizations/ksao_knowledge_graph.png`: Network of knowledge competencies
- `output/network_visualizations/ksao_skill_graph.png`: Network of skill competencies
- `output/network_visualizations/ksao_ability_graph.png`: Network of ability competencies
- `output/network_visualizations/ksao_other_graph.png`: Network of other competencies
 