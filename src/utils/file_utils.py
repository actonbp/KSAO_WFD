#!/usr/bin/env python3
"""
File utility functions for the KSAO Workforce Development toolkit.

This module provides helper functions for file operations, path management,
and data loading that are used across the project.
"""

import os
from pathlib import Path

# Define project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = DATA_DIR / "images"
TEXT_OUTPUT_DIR = DATA_DIR / "text_output"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
INTERACTIVE_VIZ_DIR = OUTPUT_DIR / "interactive_viz"
FINAL_VIZ_DIR = OUTPUT_DIR / "final_viz"
STUDY_GUIDES_DIR = OUTPUT_DIR / "study_guides"
KSAO_ANALYSIS_DIR = OUTPUT_DIR / "ksao_analysis"

def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_image_files(extension=".tif"):
    """Get all image files with the specified extension."""
    ensure_dir(IMAGES_DIR)
    return sorted([f for f in IMAGES_DIR.glob(f"*{extension}")])

def get_text_files(exclude_full=True):
    """Get all text files, optionally excluding the combined full_text.txt."""
    ensure_dir(TEXT_OUTPUT_DIR)
    if exclude_full:
        return sorted([f for f in TEXT_OUTPUT_DIR.glob("*.txt") if f.name != "full_text.txt"])
    else:
        return sorted(TEXT_OUTPUT_DIR.glob("*.txt"))

def save_output(content, filename, subdir=None):
    """Save content to an output file in the specified subdirectory."""
    if subdir:
        output_dir = OUTPUT_DIR / subdir
    else:
        output_dir = OUTPUT_DIR
    
    ensure_dir(output_dir)
    output_path = output_dir / filename
    
    if isinstance(content, str):
        with open(output_path, "w") as f:
            f.write(content)
    else:
        # Assume it's a dataframe or something with a to_csv method
        try:
            content.to_csv(output_path, index=False)
        except AttributeError:
            with open(output_path, "w") as f:
                f.write(str(content))
    
    return output_path

def load_text_data(exclude_full=True):
    """Load all text data from the text output directory."""
    text_files = get_text_files(exclude_full)
    texts = []
    
    for file_path in text_files:
        with open(file_path, "r") as f:
            texts.append(f.read())
    
    return texts, [f.name for f in text_files]