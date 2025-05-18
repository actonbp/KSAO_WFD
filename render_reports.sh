#!/bin/bash

# This script renders both Quarto reports

echo "==== Rendering KSAO Framework Report ===="
/usr/local/bin/quarto render docs/ksao_framework_report.qmd

echo "==== Rendering Thinking Trace Analysis Report ===="
/usr/local/bin/quarto render docs/thinking_trace_analysis_report.qmd

echo "==== Reports Generated ===="
echo "KSAO Framework Report: docs/ksao_framework_report.html and docs/ksao_framework_report.pdf"
echo "Thinking Trace Analysis Report: docs/thinking_trace_analysis_report.html and docs/thinking_trace_analysis_report.pdf"