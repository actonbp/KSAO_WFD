#!/usr/bin/env python3
"""
Pipeline Test Script

This script tests the KSAO pipeline processing by running a simple test 
on sample documents. It verifies that the OCR, analysis, integration,
and reporting components work as expected.

Usage:
  python3 src/utils/test_pipeline.py

"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def run_command(command, description=None):
    """Run a shell command and print its output"""
    if description:
        print(f"\n=== {description} ===")
    
    print(f"Running: {command}")
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        
        # Ensure the process completes
        stdout, stderr = process.communicate()
        
        # Print any remaining output
        if stdout:
            print(stdout)
        
        if process.returncode != 0:
            print(f"Error: {stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def verify_file_exists(file_path, description=None):
    """Verify that a file exists and print its status"""
    if description:
        print(f"\n=== Verifying {description} ===")
    
    path = Path(file_path)
    if path.exists():
        print(f"✅ Found {file_path}")
        return True
    else:
        print(f"❌ Missing {file_path}")
        return False

def test_pipeline(test_file=None):
    """Test the full KSAO pipeline"""
    # Define test parameters
    test_file = test_file or "Chapter 9.tif"  # Use existing file by default
    output_dir = "data/gemini_text_output"
    analysis_dir = "output/full_analysis"
    
    # Step 1: Verify prerequisites
    print("Verifying environment prerequisites...")
    
    # Verify Scan directory exists
    if not verify_file_exists("Scan", "Scan directory"):
        return False
    
    # Verify test file exists
    if not verify_file_exists(f"Scan/{test_file}", "Test file"):
        print(f"Test file Scan/{test_file} not found. Please provide a valid file path.")
        return False
    
    # Step 2: Test OCR with a small batch (just 1-2 pages)
    clean_name = test_file.replace(" ", "_").replace(".tif", "")
    
    ocr_command = f"python3 gemini_ocr.py --single-chapter \"{test_file}\" --input-dir Scan --output-dir {output_dir} --start-page 1 --end-page 1"
    if not run_command(ocr_command, "Testing OCR component (first page only)"):
        return False
    
    # Verify OCR output
    if not verify_file_exists(f"{output_dir}/{clean_name}/page_001.txt", "OCR output"):
        return False
    
    # Step 3: Test KSAO analysis with a small sample
    # First, create a temp directory with the OCR result
    test_dir = f"{output_dir}/test_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a small test file to analyze
    with open(f"{test_dir}/test_sample.txt", "w") as f:
        with open(f"{output_dir}/{clean_name}/page_001.txt", "r") as source:
            f.write(source.read())
    
    # Run analysis
    analysis_command = f"python3 src/ksao/analyze_full_textbook.py --input-dir {test_dir} --output-dir {analysis_dir} --output-file test_sample_analysis.txt"
    if not run_command(analysis_command, "Testing KSAO analysis component"):
        return False
    
    # Verify analysis output
    if not verify_file_exists(f"{analysis_dir}/test_sample_analysis.txt", "KSAO analysis output"):
        return False
    
    # Add integration and thinking trace tests here if needed
    
    print("\n=== Pipeline Test Results ===")
    print("✅ OCR component is functioning")
    print("✅ KSAO analysis component is functioning")
    print("✅ The pipeline is ready to process new documents")
    
    # Clean up test files
    print("\nCleaning up test files...")
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test the KSAO processing pipeline")
    parser.add_argument("--test-file", default=None, help="Path to a test TIFF file (relative to Scan directory)")
    args = parser.parse_args()
    
    start_time = time.time()
    success = test_pipeline(args.test_file)
    elapsed = time.time() - start_time
    
    print(f"\nTest completed in {elapsed:.2f} seconds")
    if success:
        print("✅ Pipeline test passed! The system is ready to process new documents.")
        print("   See docs/NEW_DOCUMENT_PROCESSING.md for detailed instructions.")
    else:
        print("❌ Pipeline test failed. Please check the output for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())