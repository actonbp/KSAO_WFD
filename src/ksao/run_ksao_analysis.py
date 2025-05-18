#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_output_directories():
    """Create all necessary output directories."""
    directories = ["output/full_analysis", "output/network_visualizations"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        print(f"Created directory: {directory}")

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import google.genai
        import networkx
        import matplotlib
        print("All required packages are installed!")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages with: pip install -r requirements.txt")
        return False

def run_full_analysis(input_dir="data/gemini_text_output", output_dir="output/full_analysis", visualize=True):
    """Run the full KSAO analysis process."""
    # Step 1: Analyze the full textbook
    analysis_file = os.path.join(output_dir, "textbook_ksao_analysis.txt")
    print("\n=== Step 1: Analyzing full textbook with Gemini 2.5 Pro Preview ===")
    try:
        subprocess.run(["python", "src/ksao/analyze_full_textbook.py", 
                       "--input-dir", input_dir, 
                       "--output-dir", output_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing textbook: {e}")
        return False
    
    if not os.path.exists(analysis_file):
        print(f"Error: Analysis file {analysis_file} not found!")
        return False
    
    # Step 2: Create network visualizations (optional)
    if visualize:
        print("\n=== Step 2: Creating KSAO network visualizations ===")
        vis_output_dir = "output/network_visualizations"
        try:
            subprocess.run(["python", "src/ksao/visualize_ksao_network.py", 
                           "--analysis-file", analysis_file,
                           "--output-dir", vis_output_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating visualizations: {e}")
            print("Analysis completed, but visualization failed.")
    
    print("\n=== Analysis Complete! ===")
    print(f"Analysis results saved to: {analysis_file}")
    if visualize:
        print(f"Network visualizations saved to: {vis_output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete KSAO analysis process")
    parser.add_argument("--input-dir", default="data/gemini_text_output", 
                       help="Directory containing chapter text files")
    parser.add_argument("--output-dir", default="output/full_analysis", 
                       help="Directory for analysis output")
    parser.add_argument("--no-visualize", action="store_true", 
                       help="Skip the visualization step")
    
    args = parser.parse_args()
    
    # Setup
    create_output_directories()
    if not check_requirements():
        return
    
    # Run the analysis
    run_full_analysis(args.input_dir, args.output_dir, not args.no_visualize)

if __name__ == "__main__":
    main() 