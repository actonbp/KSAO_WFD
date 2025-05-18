# Interactive Visualizations Directory

This directory contains interactive HTML visualizations generated from the CASAC certification materials.

## Contents

- **umap_interactive.html**: Interactive exploration tool showing the relationships between content chunks

## About These Visualizations

The interactive visualizations allow for dynamic exploration of the content relationships:

- **Points**: Each point represents a chunk of text from the source material
- **Proximity**: Points that are close together contain semantically similar content
- **Hover Information**: Hover over points to see excerpts from the source material
- **Clusters**: Colors indicate different topic clusters identified by the analysis

## How to Use These Visualizations

1. Open the HTML file in any modern web browser
2. Use mouse controls to interact:
   - Click and drag to move around the visualization
   - Scroll to zoom in and out
   - Hover over points to see detailed information
   - Use the legend to filter by cluster

## Technical Details

These visualizations are created using:
- Sentence embeddings from transformer models
- UMAP dimensionality reduction
- Clustering algorithms (DBSCAN or K-means)
- Plotly for interactive visualization

## Regenerating Visualizations

To regenerate these visualizations:

```bash
python ../../src/main.py --umap
```
