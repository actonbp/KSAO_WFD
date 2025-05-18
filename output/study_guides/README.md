# Study Guides Directory

This directory contains automatically generated study guides and learning materials based on the CASAC certification content.

## Contents

- **CASAC_Study_Guide.md**: Study guide organized by automatically detected topics

## About the Study Guides

The study guides are generated through an automated process that:

1. Extracts and processes text from the source materials
2. Uses embeddings to understand the semantic content
3. Clusters related content into coherent topics
4. Organizes the material in a logical order
5. Formats everything into readable Markdown

## How to Use the Study Guides

The study guides are provided in Markdown format, which can be:
- Viewed directly on GitHub
- Converted to PDF using tools like Pandoc
- Imported into note-taking applications
- Used as a reference when studying CASAC materials

## Customizing Study Guides

To regenerate the study guides with different parameters:

```bash
python ../../src/main.py --umap
```

You can customize the generation by editing the parameters in `src/analysis/umap_study_guide.py`.
