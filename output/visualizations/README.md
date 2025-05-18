# Visualizations Directory

This directory contains basic visualizations generated from the CASAC certification materials, including word clouds, heatmaps, and semantic maps.

## Contents

- **all_pages_wordcloud.png**: Word cloud of all text content
- **page_*_wordcloud.png**: Word clouds for individual pages
- **term_frequency_heatmap.png**: Heatmap showing term frequency across pages
- **page_embeddings.png**: t-SNE visualization of page similarities
- **umap_clusters.png**: UMAP clusters of content

## About These Visualizations

These visualizations provide different views into the text content:

- **Word Clouds**: Show the most frequent and important terms
- **Heatmaps**: Show how terms are distributed across pages
- **Embedding Visualizations**: Show semantic relationships between content

## How to Use These Visualizations

These static visualizations can be used to:
- Quickly identify key terms and themes
- Understand the distribution of topics across the material
- See relationships between different parts of the content

## Generating New Visualizations

To regenerate these visualizations:

```bash
python ../../src/main.py --basic
```
EOL < /dev/null