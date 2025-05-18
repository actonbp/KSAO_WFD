# KSAO Workforce Development Documentation

This directory contains documentation and presentations related to the KSAO Workforce Development project.

## Contents

- **gemini_approach.md**: Detailed explanation of the current Gemini-based KSAO extraction approach
- **ksao_analysis_report.qmd/html**: Report documenting the KSAO analysis results from Chapter 1
- **presentations/**: Quarto presentations about the project
  - `ksao_project_summary.qmd`: Main project overview presentation
  - Outputs can be generated in both RevealJS (HTML) and Beamer (PDF) formats
  
- **images/**: Images used in documentation

## Current Analysis Approach

The project currently uses Google's Gemini 2.5 Pro Preview model to analyze textbook content and extract KSAOs. This approach offers several advantages:

1. **Large Context Window**: Ability to process entire textbook chapters at once
2. **Comprehensive Understanding**: Better semantic understanding of complex concepts
3. **Hierarchical Classification**: Ability to identify relationships between competencies
4. **Simplified Pipeline**: Reduces the need for complex chunking and preprocessing

For more details, see `gemini_approach.md`.

## Analysis Reports

The `ksao_analysis_report.html` contains a detailed breakdown of the KSAOs extracted from Chapter 1 of the CASAC textbook, including:

- The systematic thinking process used by Gemini to analyze the text
- 30 identified KSAOs categorized by Knowledge, Skills, Abilities, and Other characteristics
- Hierarchical relationships between different KSAOs
- Developmental sequence of competency acquisition

New reports can be generated for additional chapters using the same process.

## Using Quarto for Documentation

All reports and presentations use Quarto for generation, allowing for multiple output formats.

### Prerequisites
- Quarto installed (`quarto --version` to check)
- If using PDF output: LaTeX installation (TinyTeX recommended)

### Generating Reports

To generate an HTML report:
```bash
quarto render ksao_analysis_report.qmd --to html
```

To generate a PDF report (if LaTeX is installed):
```bash
quarto render ksao_analysis_report.qmd --to pdf
```

### Generating Presentations

To generate the HTML (RevealJS) version:
```bash
quarto render presentations/ksao_project_summary.qmd --to revealjs
```

To generate the PDF (Beamer) version:
```bash
quarto render presentations/ksao_project_summary.qmd --to beamer
```

### Preview Mode

For interactive editing with live preview:
```bash
quarto preview ksao_analysis_report.qmd
```

## Customizing Templates

- Edit the YAML header in `.qmd` files to modify document settings
- For HTML reports: see [Quarto HTML options](https://quarto.org/docs/output-formats/html-basics.html)
- For PDF reports: see [Quarto PDF options](https://quarto.org/docs/output-formats/pdf-basics.html)
- For RevealJS: see [Quarto RevealJS options](https://quarto.org/docs/presentations/revealjs/)
- For Beamer: see [Quarto Beamer options](https://quarto.org/docs/presentations/beamer.html)