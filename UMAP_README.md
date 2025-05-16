# UMAP-Powered Study Guide for CASAC Training Materials

This project creates an interactive study guide from CASAC (Credentialed Alcoholism and Substance Abuse Counselor) certification materials using advanced NLP and dimensionality reduction techniques.

## Overview

Using sentence embeddings and UMAP, we transform text from the CASAC training manual into a visual learning map where:

- Similar content clusters together based on semantic meaning
- Students can navigate topics visually rather than linearly
- Key concepts emerge naturally from the content organization

## Features

1. **Semantic Text Chunking**: Breaks documents into meaningful semantic units using:
   - Sentence-based chunking
   - Paragraph detection
   - Smart text segmentation

2. **Advanced Embeddings**: Uses state-of-the-art sentence transformers (SBERT) to capture meaning beyond simple word frequency.

3. **UMAP Visualization**: Reduces high-dimensional embeddings to a 2D map that preserves both local and global structure.

4. **Automatic Topic Clustering**: Uses DBSCAN to identify natural topic clusters within the content.

5. **Interactive Visualization**: Creates an HTML visualization where students can:
   - Hover over points to see text content
   - Identify related content through proximity
   - Discover topic clusters through coloring

6. **Markdown Study Guide**: Auto-generates a structured study guide organized by topical cluster rather than document order.

## Files

- `umap_study_guide.py`: Main script to generate the UMAP study guide
- `CASAC_Study_Guide.md`: Generated study guide with content organized by topic
- `interactive_viz/umap_interactive.html`: Interactive visualization of content
- `visualizations/umap_clusters.png`: Static visualization of topic clusters

## How It Works

1. **Text Preprocessing**:
   - Loads text files from the `text_output` directory
   - Chunks text into semantic units (sentences, paragraphs)
   - Cleans and prepares text for embedding

2. **Embedding Generation**:
   - Uses SBERT to create embeddings for each text chunk
   - Creates a high-dimensional vector space where similar content is closer together

3. **Dimensionality Reduction**:
   - Applies UMAP to reduce the embedding space to 2 dimensions
   - Preserves both local similarities and global structure

4. **Clustering**:
   - Uses DBSCAN to identify natural clusters in the 2D projection
   - Labels each cluster based on key terms within its content

5. **Visualization**:
   - Creates static visualization with cluster labels
   - Builds interactive HTML visualization for exploration

6. **Study Guide Creation**:
   - Organizes content by cluster rather than document order
   - Creates a markdown study guide with section headers and content structure

## Educational Benefits

This approach offers several advantages over traditional linear learning:

1. **Topic-Based Organization**: Students can study related concepts together, regardless of where they appear in the original document.

2. **Visual Navigation**: The spatial layout helps students understand relationships between concepts.

3. **Efficient Review**: Similar content is grouped together, making it easier to review related material.

4. **Concept Discovery**: Students can discover connections between topics they might miss in linear reading.

## Usage

To generate a UMAP study guide:

```bash
python umap_study_guide.py
```

Customize the chunking approach in the script:
- For concept-level chunks: `chunk_type="sentences", chunk_size=3`
- For paragraph-level chunks: `chunk_type="paragraphs", chunk_size=1`
- For page-level analysis: `chunk_type="words", chunk_size=100`

## Further Enhancements

Future improvements could include:

1. **Quiz Generation**: Auto-generate questions for each topic cluster
2. **Flashcard Creation**: Create spaced repetition flashcards organized by topic
3. **Learning Pathways**: Generate suggested learning paths through the content
4. **Knowledge Graph**: Build explicit relationships between concepts
5. **Multi-Document Integration**: Combine multiple textbooks or resources into a unified map