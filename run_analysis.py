#!/usr/bin/env python3
"""
CASAC Text Analysis Pipeline

This script runs the full text extraction and analysis pipeline:
1. Extract text from TIF images using OCR
2. Generate basic visualizations (word clouds, term frequency)
3. Create UMAP-based study guide and visualizations
4. Generate advanced domain-enhanced visualizations
5. Create academic visualization for educational purposes
6. Generate creative data journey visualization
7. Export embeddings for use in other tools
"""

import os
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and print output."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {description}")
    print(f"{'=' * 80}")
    subprocess.run(command, shell=True)
    print(f"{'=' * 80}")
    print(f"COMPLETED: {description}")
    print(f"{'=' * 80}\n")

def main():
    parser = argparse.ArgumentParser(description="Run CASAC text analysis pipeline")
    parser.add_argument('--ocr', action='store_true', help='Run OCR to extract text from TIF images')
    parser.add_argument('--basic', action='store_true', help='Run basic visualizations (word clouds, etc.)')
    parser.add_argument('--umap', action='store_true', help='Run UMAP-based study guide generation')
    parser.add_argument('--domain', action='store_true', help='Run domain-enhanced visualizations')
    parser.add_argument('--academic', action='store_true', help='Create academic visualization')
    parser.add_argument('--creative', action='store_true', help='Generate creative data journey visualization')
    parser.add_argument('--export', action='store_true', help='Export embeddings')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # Default to all steps if none specified
    if not (args.ocr or args.basic or args.umap or args.domain or args.academic or args.creative or args.export):
        args.all = True
    
    # Create necessary directories
    Path("text_output").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)
    Path("embeddings").mkdir(exist_ok=True)
    Path("interactive_viz").mkdir(exist_ok=True)
    Path("optimal_viz").mkdir(exist_ok=True)
    Path("academic_viz").mkdir(exist_ok=True)
    Path("creative_viz").mkdir(exist_ok=True)
    Path("final_viz").mkdir(exist_ok=True)
    
    # Activate virtual environment if it exists
    if os.path.exists("venv/bin/activate"):
        activate_cmd = "source venv/bin/activate && "
    else:
        activate_cmd = ""
    
    # Step 1: Extract text from TIF images
    if args.ocr or args.all:
        run_command(f"{activate_cmd}python analyze_tif_images.py", 
                  "Extract text from TIF images and create basic visualizations")
    
    # Step 2: Run basic visualizations
    if args.basic or args.all:
        run_command(f"{activate_cmd}python text_viz_helper.py", 
                  "Generate basic text visualizations")
    
    # Step 3: Run UMAP-based study guide generation
    if args.umap or args.all:
        run_command(f"{activate_cmd}python umap_study_guide.py", 
                  "Generate UMAP study guide")
    
    # Step 4: Run domain-enhanced visualizations
    if args.domain or args.all:
        run_command(f"{activate_cmd}python domain_enhanced_viz.py", 
                  "Generate domain-enhanced visualizations")
    
    # Step 5: Create academic visualization
    if args.academic or args.all:
        run_command(f"{activate_cmd}python academic_visualization.py", 
                  "Create academic visualization for educational purposes")
    
    # Step 6: Generate creative data journey visualization
    if args.creative or args.all:
        run_command(f"{activate_cmd}python final_data_viz.py", 
                  "Generate creative data journey visualization")
    
    # Step 7: Export embeddings
    if args.export or args.all:
        run_command(f"{activate_cmd}python export_embeddings.py --chunk_type sentences --chunk_size 3", 
                  "Export embeddings for use in other tools")
    
    print("\nPipeline execution complete!")
    print("Generated files:")
    print("- text_output/: Extracted text from TIF images")
    print("- visualizations/: Basic static visualizations")
    print("- interactive_viz/: Interactive visualizations")
    print("- optimal_viz/: Domain-enhanced visualizations")
    print("- academic_viz/: Academic visualization for educational purposes")
    print("- creative_viz/: Creative data journey visualization")
    print("- final_viz/: Final visualization outputs")
    print("- embeddings/: Exported embeddings in various formats")
    print("- CASAC_Study_Guide.md: Generated study guide organized by topic")

if __name__ == "__main__":
    main()