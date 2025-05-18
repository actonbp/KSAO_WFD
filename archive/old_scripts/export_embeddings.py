import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import sent_tokenize

# Try to download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Simple sentence tokenizer as backup
def simple_sentence_tokenize(text):
    """Split text into sentences using simple rules."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]

def load_text_files(folder_path):
    """Load text files from a directory."""
    texts = []
    file_names = []
    
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.txt') and file_name != 'full_text.txt':
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                texts.append(f.read())
            file_names.append(file_name)
    
    return texts, file_names

def split_into_chunks(texts, chunk_size=75, chunk_type="words"):
    """Split texts into smaller chunks."""
    chunks = []
    chunk_metadata = []
    
    for doc_idx, text in enumerate(texts):
        if chunk_type == "words":
            # Split by words
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk.strip()) > 0:
                    chunks.append(chunk)
                    chunk_metadata.append({
                        "doc_idx": doc_idx,
                        "page": doc_idx + 1,
                        "chunk_idx": len(chunks) - 1,
                        "start_word": i,
                        "end_word": min(i + chunk_size, len(words)),
                        "chunk_type": "words"
                    })
        
        elif chunk_type == "sentences":
            # Split by sentences
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                print(f"NLTK sentence tokenization failed: {e}. Using simple tokenizer instead.")
                sentences = simple_sentence_tokenize(text)
                
            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i:i+chunk_size])
                if len(chunk.strip()) > 0:
                    chunks.append(chunk)
                    chunk_metadata.append({
                        "doc_idx": doc_idx,
                        "page": doc_idx + 1,
                        "chunk_idx": len(chunks) - 1,
                        "start_sentence": i,
                        "end_sentence": min(i + chunk_size, len(sentences)),
                        "chunk_type": "sentences"
                    })
        
        elif chunk_type == "paragraphs":
            # Split by paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            for i in range(0, len(paragraphs), chunk_size):
                chunk = "\n\n".join(paragraphs[i:i+chunk_size])
                if len(chunk.strip()) > 0:
                    chunks.append(chunk)
                    chunk_metadata.append({
                        "doc_idx": doc_idx,
                        "page": doc_idx + 1,
                        "chunk_idx": len(chunks) - 1,
                        "start_paragraph": i,
                        "end_paragraph": min(i + chunk_size, len(paragraphs)),
                        "chunk_type": "paragraphs"
                    })
    
    return chunks, chunk_metadata

def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Create embeddings using Sentence Transformers."""
    print(f"Creating embeddings with model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings with progress bar
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    return embeddings

def export_embeddings(embeddings, chunk_metadata, chunks, output_folder):
    """Export embeddings to various formats."""
    Path(output_folder).mkdir(exist_ok=True)
    
    # Create a DataFrame with embeddings and metadata
    df = pd.DataFrame({
        "text": chunks,
        "page": [meta["page"] for meta in chunk_metadata],
        "chunk_idx": [meta["chunk_idx"] for meta in chunk_metadata],
    })
    
    # Add embedding columns
    for i in range(embeddings.shape[1]):
        df[f"dim_{i}"] = embeddings[:, i]
    
    # Export as CSV
    csv_path = os.path.join(output_folder, "text_embeddings.csv")
    df.to_csv(csv_path, index=False)
    print(f"Embeddings exported to CSV: {csv_path}")
    
    # Export as JSON
    json_data = []
    for i, meta in enumerate(chunk_metadata):
        item = {
            "text": chunks[i],
            "page": meta["page"],
            "chunk_idx": meta["chunk_idx"],
            "embedding": embeddings[i].tolist()
        }
        json_data.append(item)
    
    json_path = os.path.join(output_folder, "text_embeddings.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Embeddings exported to JSON: {json_path}")
    
    # Export in projector format
    # metadata.tsv
    meta_path = os.path.join(output_folder, "metadata.tsv")
    with open(meta_path, 'w') as f:
        f.write("page\ttext\n")
        for i, chunk in enumerate(chunks):
            # Replace tabs and newlines in text
            clean_text = chunk.replace('\n', ' ').replace('\t', ' ')
            f.write(f"{chunk_metadata[i]['page']}\t{clean_text}\n")
    
    # vectors.tsv
    vectors_path = os.path.join(output_folder, "vectors.tsv")
    with open(vectors_path, 'w') as f:
        for embedding in embeddings:
            f.write('\t'.join([str(x) for x in embedding]) + '\n')
    
    print(f"Embeddings exported in TensorFlow Projector format: {meta_path} and {vectors_path}")

def main():
    parser = argparse.ArgumentParser(description="Export text embeddings in various formats.")
    parser.add_argument('--input_folder', default='text_output', help='Folder containing text files')
    parser.add_argument('--output_folder', default='embeddings', help='Folder to save embeddings')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence Transformers model name')
    parser.add_argument('--chunk_type', default='sentences', choices=['words', 'sentences', 'paragraphs'], 
                        help='Method to chunk text')
    parser.add_argument('--chunk_size', type=int, default=3, help='Size of each chunk')
    
    args = parser.parse_args()
    
    # Load text files
    texts, file_names = load_text_files(args.input_folder)
    print(f"Loaded {len(texts)} text files")
    
    # Split into chunks
    chunks, chunk_metadata = split_into_chunks(texts, chunk_size=args.chunk_size, chunk_type=args.chunk_type)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = create_embeddings(chunks, model_name=args.model)
    
    # Export embeddings
    export_embeddings(embeddings, chunk_metadata, chunks, args.output_folder)
    
    print("Done!")

if __name__ == "__main__":
    main()