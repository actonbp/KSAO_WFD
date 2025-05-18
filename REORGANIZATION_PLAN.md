# Repository Reorganization Plan

This document outlines the plan for reorganizing the KSAO_WFD repository to better focus on the KSAO (Knowledge, Skills, Abilities, and Other characteristics) extraction and workforce development goals.

## Current Structure Issues

The current repository structure has some limitations:

1. **Lack of KSAO focus**: The repository is more focused on general text analysis rather than KSAO extraction
2. **Scattered files**: Related functionality is spread across multiple scripts
3. **Unclear workflow**: The progression from raw data to final output isn't obvious
4. **Outdated files**: Some files may be superseded by newer approaches

## Proposed Structure

```
KSAO_WFD/
├── README.md
├── requirements.txt
├── config/                    # Configuration files
├── data/                      # Raw data sources
│   ├── images/                # Original document images
│   ├── Scan/                  # Additional scanned materials
│   └── text_output/           # Extracted text
├── src/                       # Source code
│   ├── extraction/            # Text extraction modules
│   ├── analysis/              # Text analysis modules
│   ├── visualization/         # Visualization modules
│   ├── ksao/                  # KSAO extraction and classification code
│   └── utils/                 # Utility functions
├── docs/                      # Documentation
│   ├── presentations/         # Quarto presentations
│   ├── images/                # Images for documentation
│   └── guides/                # Usage guides
├── notebooks/                 # Jupyter notebooks for exploration and examples
├── tests/                     # Unit tests
├── output/                    # Generated outputs
│   ├── visualizations/        # Basic visualizations
│   ├── interactive_viz/       # Interactive visualizations
│   ├── final_viz/             # Final visualizations
│   ├── study_guides/          # Generated study guides
│   └── ksao_analysis/         # KSAO analysis results
└── archive/                   # Archived files (old/deprecated)
```

## Migration Steps

1. **Create new directories**:
   - Create the new directory structure
   - Create placeholder READMEs for each directory

2. **Reorganize source code**:
   - Move extraction scripts to `src/extraction/`
   - Move analysis scripts to `src/analysis/`
   - Move visualization scripts to `src/visualization/`
   - Create new KSAO-specific code in `src/ksao/`
   - Move utility functions to `src/utils/`

3. **Reorganize data**:
   - Move images to `data/images/`
   - Move Scan directory to `data/Scan/`
   - Move text_output to `data/text_output/`

4. **Reorganize outputs**:
   - Move visualization outputs to `output/visualizations/`
   - Move interactive_viz to `output/interactive_viz/`
   - Move final_viz to `output/final_viz/`
   - Move study guides to `output/study_guides/`

5. **Identify outdated files**:
   - Review all files and identify those that are outdated or superseded
   - Move outdated files to `archive/` with explanatory notes

6. **Update imports and paths**:
   - Update all import statements in Python files
   - Update all file paths in scripts

7. **Update documentation**:
   - Update README.md with new structure
   - Create/update directory-specific READMEs

## Next Development Steps

Once the repository is reorganized, focus on these development priorities:

1. **Create KSAO extraction module**:
   - Develop specialized NLP models for extracting KSAOs from text
   - Create a schema for classifying text into K, S, A, or O categories
   - Build a validation framework for extracted KSAOs

2. **Create competency mapping tools**:
   - Develop tools to map extracted KSAOs to professional competencies
   - Create visualizations focused on competency relationships
   - Build a KSAO knowledge graph

3. **Create workforce development recommendations**:
   - Develop tools to generate evidence-based recommendations
   - Create frameworks for assessment based on KSAOs
   - Build models to compare KSAO requirements across domains

## Timeline

1. **Phase 1**: Create new directory structure and migrate files (1-2 days)
2. **Phase 2**: Update imports, paths, and fix any broken functionality (1-2 days)
3. **Phase 3**: Develop KSAO extraction module (1-2 weeks)
4. **Phase 4**: Develop competency mapping and recommendations (2-3 weeks)