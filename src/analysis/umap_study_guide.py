import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import DBSCAN, KMeans
import nltk
from nltk.tokenize import sent_tokenize
import textwrap

# Create directories
Path("embeddings").mkdir(exist_ok=True)
Path("interactive_viz").mkdir(exist_ok=True)

# Download NLTK resources for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
# Use a simpler sentence splitter if NLTK fails
def simple_sentence_tokenize(text):
    """Split text into sentences using simple rules."""
    # Split on periods, exclamation marks, and question marks followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    return [s for s in sentences if s.strip()]

def load_text_files(folder_path="text_output"):
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
    """Split texts into smaller chunks.
    
    Parameters:
    - texts: List of text documents
    - chunk_size: Size of each chunk
    - chunk_type: 'words', 'sentences', or 'paragraphs'
    
    Returns:
    - chunks: List of text chunks
    - chunk_metadata: List of dictionaries with metadata for each chunk
    """
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
                        "page": doc_idx + 1,  # 1-indexed for readability
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
            # Split by paragraphs (assuming double newlines separate paragraphs)
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

def apply_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """Apply UMAP dimensionality reduction."""
    print("Applying UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

def cluster_embeddings(umap_embeddings, method="dbscan", eps=0.5, min_samples=5, n_clusters=10):
    """Cluster the UMAP embeddings."""
    print(f"Clustering with {method}...")
    
    if method == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(umap_embeddings)
        labels = clustering.labels_
    elif method == "kmeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(umap_embeddings)
        labels = clustering.labels_
    else:
        raise ValueError("Method must be 'dbscan' or 'kmeans'")
    
    return labels

def summarize_clusters(chunks, labels):
    """Summarize the content of each cluster."""
    summaries = {}
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:  # Noise points in DBSCAN
            continue
            
        cluster_chunks = [chunks[i] for i in range(len(chunks)) if labels[i] == label]
        
        # Create a summary of the cluster
        cluster_text = " ".join(cluster_chunks)
        # Simplified summary: first 200 characters + "..."
        summary = cluster_text[:200] + "..." if len(cluster_text) > 200 else cluster_text
        
        # Add keywords based on frequency (simple approach)
        words = cluster_text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Filter out very short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [word for word, _ in keywords]
        
        summaries[label] = {
            "summary": summary,
            "keywords": keywords,
            "num_chunks": len(cluster_chunks)
        }
    
    return summaries

def create_static_visualization(umap_embeddings, chunk_metadata, labels, summaries, 
                             output_path="visualizations/umap_clusters.png"):
    """Create a static visualization of the UMAP embeddings."""
    print("Creating static visualization...")
    
    # Create a dataframe with the UMAP coordinates and metadata
    df = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "page": [meta["page"] for meta in chunk_metadata],
        "cluster": labels
    })
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot points colored by cluster
    scatter = plt.scatter(df["x"], df["y"], c=df["cluster"], cmap="tab20", 
                        alpha=0.8, s=50, edgecolors="w", linewidths=0.5)
    
    # Add cluster labels
    for label, summary in summaries.items():
        if label == -1:  # Skip noise points
            continue
            
        # Find the centroid of this cluster
        cluster_points = df[df["cluster"] == label]
        centroid_x = cluster_points["x"].mean()
        centroid_y = cluster_points["y"].mean()
        
        # Add the label text
        plt.annotate(
            f"Cluster {label}\n({summary['num_chunks']} chunks)\n{', '.join(summary['keywords'][:3])}",
            (centroid_x, centroid_y),
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7)
        )
    
    plt.title("UMAP Visualization of Document Chunks with Cluster Labels")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_interactive_visualization(umap_embeddings, chunks, chunk_metadata, labels, summaries,
                                  output_path="interactive_viz/umap_interactive.html"):
    """Create an interactive visualization using Plotly."""
    print("Creating interactive visualization...")
    
    # Create a DataFrame with all the data
    df = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "page": [meta["page"] for meta in chunk_metadata],
        "cluster": labels,
        "text": [textwrap.fill(chunk[:300] + "..." if len(chunk) > 300 else chunk, width=50) 
               for chunk in chunks]
    })
    
    # Add cluster information
    cluster_names = {}
    for label, summary in summaries.items():
        if label == -1:
            cluster_names[label] = "Noise"
        else:
            keywords = ", ".join(summary["keywords"][:3])
            cluster_names[label] = f"Cluster {label}: {keywords}"
    
    df["cluster_name"] = [cluster_names.get(label, f"Cluster {label}") for label in df["cluster"]]
    
    # Create the interactive plot
    fig = px.scatter(
        df, x="x", y="y", 
        color="cluster_name", 
        hover_data=["page", "text"],
        title="Interactive UMAP Visualization of Document Chunks"
    )
    
    # Improve the hover template
    fig.update_traces(
        hovertemplate="<b>Page %{customdata[0]}</b><br>Cluster: %{marker.color}<br><br>%{customdata[1]}<extra></extra>"
    )
    
    # Improve layout
    fig.update_layout(
        width=1000, 
        height=800,
        legend_title="Clusters",
    )
    
    # Save the figure
    fig.write_html(output_path)
    
    print(f"Interactive visualization saved to {output_path}")
    return fig

def label_chunks(chunks, chunk_metadata, labels, summaries):
    """Label each chunk with its cluster and summary information."""
    labeled_chunks = []
    
    for i, chunk in enumerate(chunks):
        cluster = labels[i]
        
        if cluster == -1:  # Noise in DBSCAN
            cluster_info = "Unclustered"
            keywords = []
        else:
            cluster_info = f"Cluster {cluster}"
            keywords = summaries[cluster]["keywords"]
        
        labeled_chunks.append({
            "chunk_text": chunk,
            "page": chunk_metadata[i]["page"],
            "cluster": cluster,
            "cluster_name": cluster_info,
            "keywords": keywords,
            "chunk_idx": i
        })
    
    return labeled_chunks

def create_study_guide(labeled_chunks, output_path="CASAC_Study_Guide.md"):
    """Create a markdown study guide from the labeled chunks."""
    print("Creating study guide...")
    
    # Group chunks by cluster
    clusters = {}
    for chunk in labeled_chunks:
        cluster = chunk["cluster"]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(chunk)
    
    # Sort clusters by size (number of chunks)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Create the study guide
    with open(output_path, "w") as f:
        f.write("# CASAC 350 HR Program Study Guide\n\n")
        f.write("This study guide is organized by topics automatically detected in the text.\n\n")
        
        for cluster, chunks in sorted_clusters:
            if cluster == -1:
                section_title = "Miscellaneous Content"
            else:
                # Get keywords for this cluster
                keywords = set()
                for chunk in chunks:
                    keywords.update(chunk["keywords"])
                keywords = list(keywords)[:5]  # Top 5 unique keywords
                
                section_title = f"Topic {cluster}: {', '.join(keywords)}"
            
            f.write(f"## {section_title}\n\n")
            
            # Sort chunks by page number
            chunks.sort(key=lambda x: x["page"])
            
            # Write each chunk
            for chunk in chunks:
                f.write(f"### From Page {chunk['page']}\n\n")
                f.write(f"{chunk['chunk_text']}\n\n")
                f.write("---\n\n")
    
    print(f"Study guide created at {output_path}")

def main():
    """Main function to generate UMAP study guide.
    
    Parameters can be configured by modifying the variables at the beginning of this function.
    """
    # Configuration parameters - adjust these as needed
    config = {
        # Input/output configuration
        "input_folder": "text_output",
        "output_folder": "visualizations",
        "interactive_folder": "interactive_viz",
        "study_guide_path": "CASAC_Study_Guide.md",
        
        # Text chunking configuration
        "chunk_type": "sentences",  # Options: "words", "sentences", "paragraphs"
        "chunk_size": 3,  # Number of units (words, sentences, paragraphs) per chunk
        
        # Embedding configuration
        "embedding_model": "all-MiniLM-L6-v2",  # Model from sentence-transformers
        
        # UMAP configuration
        "umap_n_neighbors": 15,  # Controls how local/global the projection is
        "umap_min_dist": 0.1,    # Controls compactness of embedding
        
        # Clustering configuration
        "cluster_method": "dbscan",  # Options: "dbscan", "kmeans"
        "dbscan_eps": 0.5,           # DBSCAN: max distance between points
        "dbscan_min_samples": 2,     # DBSCAN: min points to form a cluster
        "kmeans_n_clusters": 10      # KMeans: number of clusters
    }
    
    # Load text files
    texts, file_names = load_text_files(config["input_folder"])
    print(f"Loaded {len(texts)} text files")
    
    # Split into chunks
    chunks, chunk_metadata = split_into_chunks(
        texts, 
        chunk_size=config["chunk_size"], 
        chunk_type=config["chunk_type"]
    )
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = create_embeddings(chunks, model_name=config["embedding_model"])
    
    # Apply UMAP
    umap_embeddings = apply_umap(
        embeddings,
        n_neighbors=config["umap_n_neighbors"],
        min_dist=config["umap_min_dist"]
    )
    
    # Cluster the embeddings
    if config["cluster_method"] == "dbscan":
        labels = cluster_embeddings(
            umap_embeddings, 
            method="dbscan", 
            eps=config["dbscan_eps"], 
            min_samples=config["dbscan_min_samples"]
        )
    else:
        labels = cluster_embeddings(
            umap_embeddings, 
            method="kmeans", 
            n_clusters=config["kmeans_n_clusters"]
        )
    
    # Summarize clusters
    summaries = summarize_clusters(chunks, labels)
    
    # Create visualizations
    create_static_visualization(
        umap_embeddings, 
        chunk_metadata, 
        labels, 
        summaries,
        output_path=f"{config['output_folder']}/umap_clusters.png"
    )
    
    create_interactive_visualization(
        umap_embeddings, 
        chunks, 
        chunk_metadata, 
        labels, 
        summaries,
        output_path=f"{config['interactive_folder']}/umap_interactive.html"
    )
    
    # Label chunks
    labeled_chunks = label_chunks(chunks, chunk_metadata, labels, summaries)
    
    # Create study guide
    create_study_guide(labeled_chunks, output_path=config["study_guide_path"])
    
    print("Done!")

if __name__ == "__main__":
    main()