# KSAO Extraction Module

This module contains code for extracting Knowledge, Skills, Abilities, and Other characteristics (KSAOs) from text, with a focus on addiction counseling competencies.

## Components

- **extract_ksao.py**: Main KSAO extraction functionality
- (Future files to be added as development progresses)

## KSAO Framework

The KSAO framework categorizes competencies into four main categories:

1. **Knowledge**: What a person needs to know (facts, information, theories, concepts)
2. **Skills**: What a person needs to be able to do (learned capabilities, techniques)
3. **Abilities**: Innate attributes or talents (cognitive, physical, psychomotor)
4. **Other characteristics**: Additional attributes important for success (attitudes, values, traits)

## Development

This module is under active development. Key priorities include:

1. Developing NLP models for automated KSAO extraction
2. Creating a classification system for categorizing statements
3. Building validation tools for extracted KSAOs
4. Creating competency mapping visualizations

## Usage

```python
from ksao import extract_ksao

# Extract KSAOs from text
extract_ksao.main()
```

Or via the main CLI:

```bash
python ../main.py --ksao
```