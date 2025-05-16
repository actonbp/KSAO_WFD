# Embedding Analysis Process

This document outlines the steps taken to extract text from images and generate embeddings and visualizations for analysis.

## 1. OCR Process

We used the following approach for text extraction:

1. **Image Processing**: 
   - Processed 11 TIF images from `images/` folder
   - Used Tesseract OCR for text extraction
   - Saved extracted text to individual files in `text_output/` folder

2. **Text Cleaning**:
   - Removed special characters while preserving important punctuation
   - Removed excess whitespace
   - Normalized text for analysis

## 2. Embedding Generation

We used TF-IDF (Term Frequency-Inverse Document Frequency) to generate text embeddings:

1. **TF-IDF Vectorization**:
   - Created sparse vector representations of each page
   - Used parameters:
     - max_features=1000
     - stop_words='english'
     - min_df=2

2. **Dimensionality Reduction**:
   - Applied PCA for initial dimension reduction (appropriate to dataset size)
   - Applied t-SNE for final 2D visualization
   - Used perplexity parameter scaled to dataset size

## 3. Visualizations 

We created multiple visualizations to analyze the content:

1. **t-SNE Plot**:
   - 2D representation of document similarity
   - Each point represents a page
   - Proximity indicates content similarity
   - Color-coded by page number

2. **Word Clouds**:
   - Generated for the entire document
   - Generated for each individual page
   - Shows the most significant terms by frequency and relevance

3. **Term Frequency Heatmap**:
   - Shows frequency of top 20 terms across all pages
   - X-axis: terms
   - Y-axis: pages
   - Color intensity indicates frequency

4. **Top Terms Analysis**:
   - Identified the most significant terms for each page by TF-IDF score
   - Listed top 10 terms per page with their weights

## 4. Analysis Findings

The analysis revealed:

1. **Content Structure**:
   - Clear progression of topics from terminology to neurobiology to clinical patterns
   - Clustering of related content across consecutive pages

2. **Key Terminology**:
   - Consistent use of "substance use disorder" throughout
   - Specialized vocabulary in specific sections (e.g., neurobiology terms)
   - Evolution from stigmatizing to person-first language

3. **Topic Distribution**:
   - Pages 1-3: Introduction and terminology
   - Pages 4-5: Disease model and brain function
   - Pages 6-7: Risk factors and neurobiology basics
   - Pages 8-9: Brain reward systems
   - Pages 10-11: Substance use patterns and consequences

## 5. Next Steps for Enhanced Analysis

Potential enhancements to this analysis could include:

1. **Topic Modeling**:
   - Apply LDA (Latent Dirichlet Allocation) to identify latent topics
   - Use BERTopic for more advanced topic detection

2. **Advanced Embeddings**:
   - Use transformer-based embeddings (BERT, RoBERTa)
   - Apply sentence-level embeddings rather than document-level

3. **Interactive Visualizations**:
   - Create interactive dashboards with Plotly or Dash
   - Enable filtering and exploration of terms and relationships

4. **Comparative Analysis**:
   - Compare with other CASAC materials or different certification programs
   - Identify gaps or unique features in this curriculum

5. **Sentiment Analysis**:
   - Analyze the emotional tone of the material
   - Identify potentially stigmatizing or empowering language