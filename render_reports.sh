#!/bin/bash

# This script renders both Quarto reports

echo "==== Rendering KSAO Framework Report ===="
echo "Generating HTML..."
cd docs
/usr/local/bin/quarto render ksao_framework_report.qmd --output ksao_framework_report.html
echo "Generating PDF with xelatex..."
/usr/local/bin/quarto render ksao_framework_report.qmd --to pdf --output ksao_framework_report.pdf --pdf-engine=xelatex

echo "==== Rendering Thinking Trace Analysis Report ===="
echo "Generating HTML..."
/usr/local/bin/quarto render thinking_trace_analysis_report.qmd --output thinking_trace_analysis_report.html
echo "Generating PDF with xelatex..."
/usr/local/bin/quarto render thinking_trace_analysis_report.qmd --to pdf --output thinking_trace_analysis_report.pdf --pdf-engine=xelatex
cd ..

echo "==== Reports Generated ===="
echo "KSAO Framework Report: docs/ksao_framework_report.html and docs/ksao_framework_report.pdf"
echo "Thinking Trace Analysis Report: docs/thinking_trace_analysis_report.html and docs/thinking_trace_analysis_report.pdf"