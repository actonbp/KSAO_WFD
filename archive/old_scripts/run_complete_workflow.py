#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def create_output_directories():
    """Create all necessary output directories."""
    directories = [
        "data/gemini_text_output",
        "output/full_analysis",
        "output/network_visualizations",
        "docs/images"
    ]
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

def run_ocr_step(scan_dir="Scan", output_dir="data/gemini_text_output", single_chapter=None):
    """Run OCR processing with Gemini multimodal on TIFF images."""
    print("\n=== Step 1: OCR Processing with Gemini Multimodal ===")
    
    if single_chapter:
        # Process only the specified chapter
        tif_path = os.path.join(scan_dir, f"{single_chapter}.tif")
        if os.path.exists(tif_path):
            print(f"Processing single chapter: {single_chapter}")
            try:
                subprocess.run(["python3", "gemini_ocr.py", 
                              "--single-chapter", f"{single_chapter}.tif", 
                              "--input-dir", scan_dir,
                              "--output-dir", output_dir], check=True)
                print(f"Successfully processed OCR for {single_chapter}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error processing OCR for {single_chapter}: {e}")
                return False
        else:
            print(f"Error: TIF file for {single_chapter} not found in {scan_dir}")
            return False
    else:
        # Process all chapters
        try:
            subprocess.run(["python3", "gemini_ocr.py", 
                           "--process-all",
                           "--input-dir", scan_dir,
                           "--output-dir", output_dir], check=True)
            print("Successfully processed OCR for all chapters")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error processing OCR: {e}")
            return False

def run_ksao_extraction(text_output_dir="data/gemini_text_output", 
                       analysis_output_dir="output/full_analysis", 
                       single_chapter=None):
    """Extract KSAOs for each chapter with Gemini."""
    print("\n=== Step 2: KSAO Extraction with Gemini ===")
    
    # Get a list of all chapter text files
    text_path = Path(text_output_dir)
    full_text_files = [f for f in text_path.glob("*_full.txt") if not f.name == "complete_book.txt"]
    
    if not full_text_files:
        print(f"No chapter text files found in {text_output_dir}")
        return False
    
    success_count = 0
    # Process each chapter separately
    for text_file in full_text_files:
        chapter_name = text_file.stem.replace("_full", "")
        
        if single_chapter and single_chapter.lower() not in chapter_name.lower():
            print(f"Skipping KSAO analysis for {chapter_name} (not the selected chapter)")
            continue
        
        print(f"Analyzing KSAO for {chapter_name}...")
        
        # Create temp directory for this chapter's text to isolate it for analysis
        temp_dir = Path(f"{text_output_dir}/temp_{chapter_name}")
        temp_dir.mkdir(exist_ok=True)
        
        # Copy just this chapter file to the temp directory
        import shutil
        shutil.copy(text_file, temp_dir / text_file.name)
        
        # Run KSAO analysis on this chapter
        output_file = f"{chapter_name}_ksao_analysis.txt"
        try:
            subprocess.run(["python3", "src/ksao/analyze_full_textbook.py", 
                           "--input-dir", str(temp_dir),
                           "--output-dir", analysis_output_dir,
                           "--output-file", output_file], check=True)
            print(f"KSAO analysis completed for {chapter_name}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error analyzing {chapter_name}: {e}")
            continue
        
        # Remove temp directory
        shutil.rmtree(temp_dir)
        
        # Brief pause to avoid API rate limits
        time.sleep(2)
    
    return success_count > 0

def run_ksao_integration(analysis_output_dir="output/full_analysis"):
    """Integrate all KSAO analyses into a comprehensive framework."""
    print("\n=== Step 3: Integrating KSAO Analyses ===")
    
    # Check if we have multiple chapter analyses to integrate
    analysis_path = Path(analysis_output_dir)
    analysis_files = [f for f in analysis_path.glob("*_ksao_analysis.txt") 
                     if not f.name.endswith("_thinking_process.txt") 
                     and not f.name.endswith("_combined_text.txt")
                     and not "integrated" in f.name]
    
    if len(analysis_files) < 2:
        print("Need at least two chapter analyses to perform integration")
        if len(analysis_files) == 1:
            print("Only one chapter analysis found - skipping integration step")
            return True
        else:
            print("No chapter analyses found")
            return False
    
    try:
        print(f"Integrating {len(analysis_files)} chapter analyses...")
        subprocess.run(["python3", "src/ksao/integrate_ksao_analyses.py", 
                       "--input-dir", analysis_output_dir, 
                       "--output-dir", analysis_output_dir], check=True)
        print("KSAO integration completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error integrating KSAO analyses: {e}")
        print("Individual chapter analyses are still available")
        return False

def generate_quarto_report(analysis_output_dir="output/full_analysis"):
    """Generate a Quarto report with the analysis results."""
    print("\n=== Step 4: Generating Final Report ===")
    
    # Get paths to the integrated analysis if it exists, otherwise use the first chapter analysis
    analysis_path = Path(analysis_output_dir)
    integrated_file = analysis_path / "integrated_ksao_framework.txt"
    
    if not integrated_file.exists():
        # Find the first chapter analysis file
        analysis_files = [f for f in analysis_path.glob("*_ksao_analysis.txt") 
                         if not f.name.endswith("_thinking_process.txt") 
                         and not f.name.endswith("_combined_text.txt")]
        
        if not analysis_files:
            print("No analysis files found to include in the report")
            return False
        
        analysis_file = analysis_files[0]
    else:
        analysis_file = integrated_file
    
    # Update the report template with current analysis information
    try:
        # Get the latest date from file modification time
        analysis_date = time.strftime("%B %d, %Y", time.localtime(os.path.getmtime(analysis_file)))
        
        # Read the existing Quarto report template
        with open("docs/ksao_analysis_report.qmd", 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Update the date in the template
        if "date: today" in template:
            template = template.replace("date: today", f'date: "{analysis_date}"')
        
        # Update the template with current analysis information
        with open("docs/ksao_analysis_report.qmd", 'w', encoding='utf-8') as f:
            f.write(template)
        
        # Render the Quarto report
        try:
            subprocess.run(["/opt/homebrew/bin/quarto", "render", "docs/ksao_analysis_report.qmd"], check=True)
            print("Quarto report generated successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error rendering Quarto report: {e}")
            print("You may need to render the report manually with 'quarto render docs/ksao_analysis_report.qmd'")
            return False
    except Exception as e:
        print(f"Error updating Quarto template: {e}")
        return False

def run_thinking_analysis(analysis_output_dir="output/full_analysis"):
    """Analyze thinking traces from all chapters to create a meta-analysis."""
    print("\n=== Step 4: Analyzing Thinking Traces Across Chapters ===")
    
    try:
        print("Analyzing thinking traces from all chapters...")
        subprocess.run(["python3", "src/ksao/analyze_thinking_traces.py", 
                       "--input-dir", analysis_output_dir, 
                       "--output-dir", analysis_output_dir], check=True)
        print("Thinking trace analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing thinking traces: {e}")
        print("Individual thinking traces are still available")
        return False

def run_complete_workflow(scan_dir="Scan", text_output_dir="data/gemini_text_output", 
                        analysis_output_dir="output/full_analysis", single_chapter=None):
    """Run the complete KSAO analysis workflow."""
    print("\n=== Starting Complete KSAO Analysis Workflow ===")
    
    # Step 1: OCR with Gemini multimodal
    ocr_success = run_ocr_step(scan_dir, text_output_dir, single_chapter)
    if not ocr_success:
        print("OCR processing failed - workflow stopped")
        return False
    
    # Step 2: Extract KSAOs for each chapter
    extraction_success = run_ksao_extraction(text_output_dir, analysis_output_dir, single_chapter)
    if not extraction_success:
        print("KSAO extraction failed - workflow stopped")
        return False
    
    # Step 3: Integrate all KSAO analyses (skip if processing only one chapter)
    if not single_chapter:
        integration_success = run_ksao_integration(analysis_output_dir)
        if not integration_success:
            print("Warning: KSAO integration failed but individual analyses are available")
    
    # Step 4: Analyze thinking traces across chapters (skip if processing only one chapter)
    if not single_chapter:
        thinking_success = run_thinking_analysis(analysis_output_dir)
        if not thinking_success:
            print("Warning: Thinking trace analysis failed but individual thinking traces are available")
    
    # Step 5: Generate Quarto report
    report_success = generate_quarto_report(analysis_output_dir)
    
    print("\n=== Complete KSAO Analysis Workflow Finished! ===")
    print(f"OCR text files saved to: {text_output_dir}")
    print(f"KSAO analyses saved to: {analysis_output_dir}")
    if report_success:
        print(f"Final report saved to: docs/ksao_analysis_report.pdf and docs/ksao_analysis_report.html")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete KSAO analysis workflow")
    parser.add_argument("--scan-dir", default="Scan", 
                       help="Directory containing TIFF scan files")
    parser.add_argument("--text-output-dir", default="data/gemini_text_output", 
                       help="Directory for extracted text output")
    parser.add_argument("--analysis-output-dir", default="output/full_analysis", 
                       help="Directory for analysis output")
    parser.add_argument("--single-chapter", default=None, 
                       help="Process only a specific chapter (e.g., 'Chapter 1')")
    parser.add_argument("--skip-ocr", action="store_true",
                       help="Skip OCR step (use existing text files)")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip KSAO extraction step (use existing analyses)")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip KSAO integration step")
    parser.add_argument("--skip-thinking", action="store_true",
                       help="Skip thinking trace analysis step")
    parser.add_argument("--only-thinking", action="store_true",
                       help="Only run the thinking trace analysis step")
    parser.add_argument("--only-report", action="store_true",
                       help="Only generate the Quarto report from existing analyses")
    
    args = parser.parse_args()
    
    # Setup
    create_output_directories()
    if not check_requirements():
        return 1
    
    # Run only the report generation if requested
    if args.only_report:
        generate_quarto_report(args.analysis_output_dir)
        return 0
    
    # Run only the thinking trace analysis if requested
    if args.only_thinking:
        run_thinking_analysis(args.analysis_output_dir)
        return 0
    
    # Run specific steps as requested
    if args.skip_ocr and args.skip_extraction and args.skip_integration and args.skip_thinking:
        print("All processing steps are skipped. Nothing to do.")
        return 0
    
    if not args.skip_ocr:
        ocr_success = run_ocr_step(args.scan_dir, args.text_output_dir, args.single_chapter)
        if not ocr_success:
            return 1
    
    if not args.skip_extraction:
        extraction_success = run_ksao_extraction(args.text_output_dir, args.analysis_output_dir, args.single_chapter)
        if not extraction_success:
            return 1
    
    if not args.skip_integration and not args.single_chapter:
        integration_success = run_ksao_integration(args.analysis_output_dir)
    
    if not args.skip_thinking and not args.single_chapter:
        thinking_success = run_thinking_analysis(args.analysis_output_dir)
    
    # Always generate the report unless specifically skipped
    generate_quarto_report(args.analysis_output_dir)
    
    return 0

if __name__ == "__main__":
    exit(main())