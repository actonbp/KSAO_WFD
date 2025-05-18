#!/bin/bash
# Script to reorganize the KSAO_WFD repository structure

echo "=== Starting KSAO_WFD Repository Reorganization ==="

# Create necessary directories
echo "Creating directory structure..."
mkdir -p src/ksao src/extraction src/analysis src/visualization src/utils
mkdir -p data/gemini_text_output data/Scan data/images
mkdir -p output/full_analysis output/network_visualizations
mkdir -p archive/old_scripts

# Move scripts to src/ subdirectories
echo "Moving scripts to src/ directory..."
cp analyze_full_textbook.py src/ksao/
cp visualize_ksao_network.py src/ksao/
cp run_ksao_analysis.py src/ksao/
cp gemini_ocr.py src/extraction/

# Move data files
echo "Moving data files..."
cp -r gemini_text_output/* data/gemini_text_output/ 2>/dev/null || echo "No files found to copy"

# Move old scripts to archive
echo "Archiving old scripts..."
mv analyze_tif_images.py text_viz_helper.py umap_study_guide.py optimal_visualization.py \
   ocr_process.py domain_enhanced_viz.py final_visualization.py academic_visualization.py \
   export_embeddings.py final_data_viz.py create_raw_data_viz.py create_improved_viz.py \
   run_analysis.py archive/old_scripts/ 2>/dev/null || echo "Some files not found for archiving"

# Create __init__.py files for Python package structure
echo "Creating Python package structure..."
touch src/__init__.py
touch src/ksao/__init__.py
touch src/extraction/__init__.py
touch src/analysis/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py

echo "=== Repository reorganization complete ==="
echo "To use the new structure, run: python src/main.py analyze" 