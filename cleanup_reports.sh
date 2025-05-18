#!/bin/bash

# This script cleans up redundant report files and organizes them properly

echo "==== Cleaning up report files ===="

# Create archive directory for old reports if it doesn't exist
mkdir -p archive/old_reports

# Move old report files to archive
echo "Moving old report files to archive/old_reports..."

# First, handle the ksao_analysis files (completely archived)
if [ -f docs/ksao_analysis_report.html ]; then
  mv docs/ksao_analysis_report.html archive/old_reports/
fi

if [ -f docs/ksao_analysis_report.qmd ]; then
  mv docs/ksao_analysis_report.qmd archive/old_reports/
fi

if [ -f docs/ksao_analysis_report.tex ]; then
  mv docs/ksao_analysis_report.tex archive/old_reports/
fi

if [ -d docs/ksao_analysis_report_files ]; then
  mv docs/ksao_analysis_report_files archive/old_reports/
fi

# Now handle redundant files for enhanced templates
# Archive _enhanced duplicates since we've copied their contents to the main files
if [ -f docs/ksao_framework_report_enhanced.qmd ]; then
  mv docs/ksao_framework_report_enhanced.qmd archive/old_reports/
fi

if [ -d docs/ksao_framework_report_enhanced_files ]; then
  mv docs/ksao_framework_report_enhanced_files archive/old_reports/
fi

if [ -f docs/thinking_trace_analysis_report_enhanced.qmd ]; then
  mv docs/thinking_trace_analysis_report_enhanced.qmd archive/old_reports/
fi

if [ -d docs/thinking_trace_analysis_report_enhanced_files ]; then
  mv docs/thinking_trace_analysis_report_enhanced_files archive/old_reports/
fi

# Archive old report files
if [ -d docs/ksao_framework_report_files ]; then
  mv docs/ksao_framework_report_files archive/old_reports/
fi

if [ -d docs/thinking_trace_analysis_report_files ]; then
  mv docs/thinking_trace_analysis_report_files archive/old_reports/
fi

echo "==== Cleanup complete ===="
echo "Old report files moved to archive/old_reports/"
echo "Only necessary report files remain in docs/ directory"