# KSAO Workforce Development - CASAC Text Analysis

A project for the Binghamton University Workforce Development Group analyzing CASAC (Credentialed Alcoholism and Substance Abuse Counselor) certification materials.

## Project Overview

This project analyzes text extracted from CASAC 350 HR Program materials to understand key themes, terminology, and content relationships. It uses advanced NLP techniques to create visual study guides and interactive learning tools.

## Components

1. **Text Extraction & Basic Analysis**
   - OCR processing of document images 
   - Word frequency analysis
   - Word clouds and heatmaps

2. **Advanced Semantic Analysis**
   - Sentence embeddings using transformer models
   - Dimensionality reduction with UMAP
   - Topic clustering with DBSCAN/K-means

3. **Interactive Study Tools**
   - Topic-based study guide
   - Interactive visualization of content relationships
   - Exportable embeddings for other applications

## Key Features

- **Semantic Understanding**: Uses transformer models to understand meaning, not just keywords
- **Visual Learning Maps**: Shows relationships between concepts through spatial organization
- **Automatic Topic Detection**: Identifies natural clusters of related content
- **Interactive Exploration**: Allows students to navigate content visually

## Repository Structure

- `images/`: TIF images of document pages
- `text_output/`: Extracted text from each page
- `visualizations/`: Basic generated visualizations
- `interactive_viz/`: Interactive HTML visualizations
- `optimal_viz/`: Domain-enhanced visualizations
- `academic_viz/`: Educational visualizations for academic purposes
- `creative_viz/`: Creative visualizations showing the data analysis journey
- `final_viz/`: Final comprehensive visualizations with cluster details
- `embeddings/`: Exportable embeddings in various formats

### Scripts

- `analyze_tif_images.py`: OCR and basic text analysis
- `text_viz_helper.py`: Helper functions for text visualization
- `umap_study_guide.py`: Advanced semantic analysis and study guide generation
- `domain_enhanced_viz.py`: Domain-specific enhancement of visualizations
- `academic_visualization.py`: Academic-focused visualization for educational contexts
- `final_data_viz.py`: Creative data journey visualization showing the analysis process
- `export_embeddings.py`: Exports embeddings for use in other tools
- `run_analysis.py`: All-in-one script to run the entire pipeline

### Output Files

- `CASAC_Study_Guide.md`: Study guide organized by topic
- `visualizations/umap_clusters.png`: Visual map of content relationships
- `interactive_viz/umap_interactive.html`: Interactive exploration tool
- `optimal_viz/domain_visualization.png`: Domain-enhanced visualization
- `academic_viz/academic_casac_map.png`: Academic visualization for educational purposes
- `creative_viz/casac_data_journey.png`: Creative visualization of the data analysis journey
- `final_viz/casac_learning_map.png`: Final visualization with cluster wordclouds

## Getting Started

### Prerequisites

- Python 3.9+
- Tesseract OCR
- Virtual environment (recommended)

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

4. Install Tesseract OCR:
   - Mac: `brew install tesseract`
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `apt-get install tesseract-ocr`

### Usage

Run the complete analysis pipeline:
```
python run_analysis.py --all
```

Or run individual steps:
```
python run_analysis.py --ocr       # Extract text using OCR
python run_analysis.py --basic     # Generate basic visualizations
python run_analysis.py --umap      # Create UMAP study guide
python run_analysis.py --domain    # Generate domain-enhanced visualizations
python run_analysis.py --academic  # Create academic visualization
python run_analysis.py --creative  # Generate creative data journey visualization
python run_analysis.py --export    # Export embeddings
```

You can also combine multiple options:
```
python run_analysis.py --ocr --basic --umap  # Run the first 3 steps only
```

## Customization

### UMAP Study Guide Parameters

Edit the configuration in `umap_study_guide.py` to customize:
- Chunking method (sentences, paragraphs, words)
- Embedding model
- UMAP parameters
- Clustering approach

### Embedding Export Options

Run with custom parameters:
```
python export_embeddings.py --chunk_type paragraphs --chunk_size 1 --model all-mpnet-base-v2
```

## References

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Interactive Visualization with Plotly](https://plotly.com/python/)

## Documentation

- `UMAP_README.md`: Detailed information about the UMAP study guide approach
- `analysis_summary.md`: Summary of findings from text analysis
- `embedding_analysis_steps.md`: Documentation of the analysis process

## Future Improvements

1. **Quiz Generation**: Auto-generate questions for each topic cluster
2. **Flashcard Creation**: Create spaced repetition flashcards organized by topic
3. **Learning Pathways**: Generate suggested learning paths through the content
4. **Knowledge Graph**: Build explicit relationships between concepts
5. **Multi-Document Integration**: Combine multiple textbooks into a unified map