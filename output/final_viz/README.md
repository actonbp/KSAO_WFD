# Final Visualizations Directory

This directory contains the final, polished visualizations generated from the CASAC certification materials analysis.

## Contents

- **casac_learning_map.html**: Interactive visualization of the CASAC learning materials organized by domains and topics
- **casac_learning_map.png**: Static version of the learning map for inclusion in documents and presentations
- **wordcloud_cluster_*.png**: Word clouds for each identified content cluster showing key terms and concepts

## Visualization Approach

The visualizations in this directory are the culmination of our text analysis and embedding process:

1. Text is extracted from certification materials
2. Content is divided into meaningful chunks
3. Advanced sentence embeddings are generated
4. UMAP dimensionality reduction is applied
5. Clustering algorithms identify related topics
6. Domain knowledge is integrated for context

## How to Use These Visualizations

### HTML Interactive Map

Open `casac_learning_map.html` in any modern web browser to:
- Explore the relationship between different topics
- Hover over points to see excerpts from the source material
- Filter by domain to focus on specific knowledge areas

### Cluster Word Clouds

The word clouds provide a visual summary of the key concepts in each identified cluster:
- Larger words appear more frequently or are more significant in the cluster
- Colors are used for visual differentiation
- Each cloud focuses on a distinct competency area

## Generating New Visualizations

To regenerate these visualizations or create new ones with different parameters:

```bash
python ../../src/main.py --final
```
