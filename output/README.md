# Output Directory

This directory contains all generated outputs from the KSAO Workforce Development project.

## Directory Structure

- `full_analysis/`: Contains results from the Gemini-based KSAO analysis
  - `textbook_ksao_analysis.txt`: The complete KSAO analysis from Gemini
  - `textbook_ksao_analysis_combined_text.txt`: Combined text that was sent to Gemini
  
- `network_visualizations/`: Contains network graph visualizations of KSAOs
  - `ksao_network_data.json`: JSON data representing the KSAO network
  - `ksao_network_graph.png`: Standard network graph visualization
  - `ksao_hierarchical_graph.png`: Hierarchical layout of KSAO relationships
  - `ksao_knowledge_graph.png`: Network of Knowledge competencies
  - `ksao_skill_graph.png`: Network of Skill competencies
  - `ksao_ability_graph.png`: Network of Ability competencies
  - `ksao_other_graph.png`: Network of Other competencies

## Output Formats

### KSAO Analysis Format

The `textbook_ksao_analysis.txt` file contains a structured analysis with:

1. Identified KSAOs and their descriptions
2. Classification (Knowledge, Skill, Ability, or Other)
3. Specificity level (general or specialized)
4. Related O*NET occupational categories
5. Stability/malleability classification
6. Explicit/tacit orientation
7. Prerequisites or developmental relationships
8. Hierarchical structure information

### Network Visualization Format

The `ksao_network_data.json` file follows this structure:

```json
{
    "nodes": [
        {
            "id": 1,
            "name": "Example KSAO Name",
            "classification": "Knowledge|Skill|Ability|Other",
            "level": "General|Specialized"
        },
        ...
    ],
    "links": [
        {
            "source": 1,
            "target": 2,
            "relationship": "prerequisite|dimension|foundation"
        },
        ...
    ]
}
```

## Viewing Outputs

- Text files can be viewed with any text editor
- PNG visualization files can be viewed with any image viewer
- JSON files can be viewed with a text editor or specialized JSON viewer
- To regenerate visualizations, run: `python src/main.py visualize`