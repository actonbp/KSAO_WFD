#!/usr/bin/env python3
"""
KSAO Extraction Module

This module contains functionality for extracting Knowledge, Skills, Abilities, and Other 
characteristics (KSAOs) from text, specifically focused on addiction counseling materials.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define KSAO categories and keywords to help with classification
KSAO_CATEGORIES = {
    "Knowledge": {
        "description": "Factual information and understanding",
        "keywords": [
            "know", "understand", "comprehend", "recognize", "identify", 
            "awareness", "theory", "concept", "principle", "fact",
            "information", "knowledge", "background", "education"
        ],
        "patterns": [
            r"knowledge of \w+",
            r"understanding of \w+",
            r"familiarity with \w+",
            r"background in \w+",
            r"education in \w+"
        ]
    },
    "Skills": {
        "description": "Learned abilities and techniques",
        "keywords": [
            "skill", "ability", "technique", "method", "approach", 
            "proficiency", "competence", "capability", "aptitude",
            "conduct", "perform", "execute", "implement", "apply"
        ],
        "patterns": [
            r"ability to \w+",
            r"skilled in \w+",
            r"proficient at \w+",
            r"can \w+",
            r"capable of \w+"
        ]
    },
    "Abilities": {
        "description": "Inherent traits or capabilities",
        "keywords": [
            "capacity", "potential", "aptitude", "talent", "faculty",
            "capability", "innate", "inherent", "natural", "inborn",
            "disposition", "inclination", "propensity", "tendency"
        ],
        "patterns": [
            r"capacity for \w+",
            r"natural ability to \w+",
            r"inherent capability to \w+",
            r"aptitude for \w+",
            r"talent for \w+"
        ]
    },
    "Other": {
        "description": "Additional characteristics for success",
        "keywords": [
            "attribute", "quality", "trait", "characteristic", "feature",
            "temperament", "disposition", "attitude", "outlook", "mindset",
            "value", "ethic", "principle", "standard", "behavior"
        ],
        "patterns": [
            r"character trait of \w+",
            r"quality of \w+",
            r"disposition towards \w+",
            r"attitude of \w+",
            r"commitment to \w+"
        ]
    }
}

def load_text_data(input_dir="../data/text_output"):
    """Load text data from the processed text files."""
    text_files = sorted(Path(input_dir).glob("*.txt"))
    texts = []
    
    for file_path in text_files:
        if file_path.name != "full_text.txt":  # Skip the combined file
            with open(file_path, "r") as f:
                texts.append(f.read())
    
    return texts

def extract_candidate_ksaos(texts):
    """Extract candidate KSAO statements from text."""
    # This is a placeholder function that will eventually implement
    # sophisticated NLP to extract KSAO statements
    
    # For now, just use simple rule-based extraction
    candidate_ksaos = []
    
    for text in texts:
        # Split into sentences
        sentences = text.split(". ")
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Check for KSAO indicators
            for category, info in KSAO_CATEGORIES.items():
                # Check for keywords
                has_keyword = any(keyword in sentence.lower() for keyword in info["keywords"])
                
                # Check for patterns
                has_pattern = any(re.search(pattern, sentence.lower()) for pattern in info["patterns"])
                
                if has_keyword or has_pattern:
                    candidate_ksaos.append({
                        "text": sentence,
                        "category": category,
                        "confidence": 0.5  # Placeholder confidence score
                    })
    
    return pd.DataFrame(candidate_ksaos)

def classify_ksaos(candidate_ksaos):
    """Classify statements as Knowledge, Skills, Abilities, or Other."""
    # This is a placeholder for a more sophisticated classification model
    # In a real implementation, this would use a trained classifier
    
    # For now, just use the preliminary category from extraction
    return candidate_ksaos

def save_ksaos(ksaos, output_path="../output/ksao_analysis/ksao_extraction.csv"):
    """Save extracted KSAOs to a CSV file."""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    ksaos.to_csv(output_path, index=False)
    print(f"Saved {len(ksaos)} KSAO statements to {output_path}")

def main():
    """Main entry point for KSAO extraction."""
    print("Loading text data...")
    texts = load_text_data()
    
    print("Extracting candidate KSAO statements...")
    candidate_ksaos = extract_candidate_ksaos(texts)
    
    print("Classifying KSAO statements...")
    classified_ksaos = classify_ksaos(candidate_ksaos)
    
    print("Saving results...")
    save_ksaos(classified_ksaos)
    
    print("KSAO extraction complete!")

if __name__ == "__main__":
    main()