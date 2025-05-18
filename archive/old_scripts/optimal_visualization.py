import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
import textwrap
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import umap

# Create directories
Path("optimal_viz").mkdir(exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def simple_sentence_tokenize(text):
    """Split text into sentences using simple rules."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
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

def extract_paragraphs(texts):
    """Extract paragraphs with better boundary detection."""
    all_paragraphs = []
    paragraph_metadata = []
    
    for doc_idx, text in enumerate(texts):
        # Better paragraph splitting that preserves headers and lists
        raw_paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.|\•|\-\s)', text)
        
        # Clean and filter paragraphs
        paragraphs = []
        for p in raw_paragraphs:
            p = p.strip()
            if len(p) > 20:  # Ensure paragraph has meaningful content
                paragraphs.append(p)
                
        # Add each paragraph with metadata
        for i, para in enumerate(paragraphs):
            all_paragraphs.append(para)
            paragraph_metadata.append({
                "doc_idx": doc_idx,
                "page": doc_idx + 1,
                "paragraph_idx": i,
                "paragraph_num": i + 1,
                "total_paragraphs": len(paragraphs)
            })
    
    return all_paragraphs, paragraph_metadata

def create_overlapping_chunks(paragraphs, paragraph_metadata, window_size=2, overlap=1):
    """Create overlapping chunks to maintain context between paragraphs."""
    chunks = []
    chunk_metadata = []
    
    # Group by document to prevent overlap between documents
    doc_indices = {}
    for i, meta in enumerate(paragraph_metadata):
        doc_idx = meta["doc_idx"]
        if doc_idx not in doc_indices:
            doc_indices[doc_idx] = []
        doc_indices[doc_idx].append(i)
    
    # Create overlapping windows for each document
    for doc_idx, indices in doc_indices.items():
        for i in range(0, len(indices), window_size - overlap):
            if i + window_size <= len(indices):
                window_indices = indices[i:i+window_size]
            else:
                window_indices = indices[i:]
            
            # Create chunk from this window
            window_paragraphs = [paragraphs[idx] for idx in window_indices]
            chunk_text = "\n\n".join(window_paragraphs)
            chunks.append(chunk_text)
            
            # Create metadata for this chunk
            start_para = paragraph_metadata[window_indices[0]]["paragraph_num"]
            end_para = paragraph_metadata[window_indices[-1]]["paragraph_num"]
            page = paragraph_metadata[window_indices[0]]["page"]
            
            chunk_metadata.append({
                "doc_idx": doc_idx,
                "page": page,
                "start_para": start_para,
                "end_para": end_para,
                "chunk_indices": window_indices
            })
    
    return chunks, chunk_metadata

def create_embeddings(chunks, model_name="all-mpnet-base-v2"):
    """Create embeddings using a stronger Sentence Transformers model."""
    print(f"Creating embeddings with model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings with progress bar
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    return embeddings

def apply_hierarchical_clustering(embeddings, n_clusters=8):
    """Apply hierarchical clustering to the embeddings."""
    print(f"Applying hierarchical clustering with {n_clusters} clusters...")
    
    # Create similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Apply hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average',
        distance_threshold=None
    )
    # Convert similarity to distance
    labels = clustering.fit_predict(1 - similarity_matrix)
    
    return labels

def apply_umap_with_parameters(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'):
    """Apply UMAP with custom parameters."""
    print(f"Applying UMAP with parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

def identify_central_concepts(embeddings, labels):
    """Identify central concepts within each cluster."""
    centrality_scores = []
    
    for i in range(len(embeddings)):
        # Calculate average similarity to other points in same cluster
        cluster = labels[i]
        cluster_indices = np.where(labels == cluster)[0]
        
        if len(cluster_indices) <= 1:
            centrality_scores.append(0)
            continue
        
        # Calculate similarities to other points in cluster
        similarities = []
        for j in cluster_indices:
            if i != j:
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                similarities.append(sim)
        
        # Average similarity is the centrality score
        if similarities:
            centrality_scores.append(np.mean(similarities))
        else:
            centrality_scores.append(0)
    
    return np.array(centrality_scores)

def extract_keywords(chunks, labels):
    """Extract keywords that characterize each cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Group chunks by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[i])
    
    # Extract keywords for each cluster
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        max_df=0.7,
        min_df=2
    )
    
    # Fit on all chunks
    vectorizer.fit(chunks)
    
    # Extract top keywords for each cluster
    cluster_keywords = {}
    for label, cluster_chunks in clusters.items():
        # Combine all text in this cluster
        cluster_text = " ".join(cluster_chunks)
        
        # Get TF-IDF scores
        tfidf = vectorizer.transform([cluster_text])
        
        # Get top keywords
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        
        # Sort by score and take top 10
        indices = np.argsort(scores)[::-1][:10]
        top_keywords = [(feature_names[i], scores[i]) for i in indices]
        
        cluster_keywords[label] = top_keywords
    
    return cluster_keywords

def create_advanced_visualization(umap_embeddings, chunk_metadata, labels, centrality_scores, 
                               cluster_keywords, chunks, output_path="optimal_viz/advanced_clustering.html"):
    """Create an advanced interactive visualization with plotly."""
    print("Creating advanced visualization...")
    
    # Prepare data for visualization
    df = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "page": [meta["page"] for meta in chunk_metadata],
        "cluster": labels,
        "centrality": centrality_scores,
        "text": [textwrap.shorten(chunk, width=300, placeholder="...") for chunk in chunks]
    })
    
    # Add cluster names based on keywords
    cluster_names = {}
    for label, keywords in cluster_keywords.items():
        # Get top 3 keywords
        top_words = [word for word, score in keywords[:3]]
        cluster_names[label] = f"Topic {label}: {', '.join(top_words)}"
    
    df["cluster_name"] = df["cluster"].map(cluster_names)
    
    # Create the visualization
    fig = px.scatter(
        df, 
        x="x", 
        y="y", 
        color="cluster_name",
        size="centrality",
        hover_data=["page", "text"],
        title="CASAC Content Map: Hierarchical Clustering of Key Concepts",
        height=800,
        width=1000,
        size_max=25,
        opacity=0.8
    )
    
    # Add cluster labels at centroids
    for label, keywords in cluster_keywords.items():
        cluster_df = df[df["cluster"] == label]
        centroid_x = cluster_df["x"].mean()
        centroid_y = cluster_df["y"].mean()
        
        # Get top keywords as string
        top_keywords = ", ".join([word for word, score in keywords[:5]])
        
        # Add annotation for cluster
        fig.add_annotation(
            x=centroid_x,
            y=centroid_y,
            text=f"Topic {label}:<br>{top_keywords}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            bgcolor="#ffffff",
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            font=dict(size=12, color="black")
        )
    
    # Improve the hover template
    fig.update_traces(
        hovertemplate="<b>Page %{customdata[0]}</b><br>Topic: %{marker.color}<br><br>%{customdata[1]}<extra></extra>"
    )
    
    # Layout improvements
    fig.update_layout(
        legend_title="Topic Clusters",
        showlegend=True,
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    # Save the figure
    fig.write_html(output_path)
    print(f"Interactive visualization saved to {output_path}")
    
    # Also create a PNG version
    png_path = output_path.replace(".html", ".png")
    fig.write_image(png_path, width=1200, height=800, scale=2)
    print(f"Static image saved to {png_path}")
    
    return fig

def create_networkx_visualization(embeddings, umap_embeddings, labels, cluster_keywords, 
                               chunk_metadata, chunks, centrality_scores,
                               output_path="optimal_viz/knowledge_graph.png"):
    """Create a network visualization showing relationships between concepts."""
    print("Creating network visualization...")
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for each chunk
    for i in range(len(chunks)):
        # Truncate long text
        short_text = textwrap.shorten(chunks[i], width=100, placeholder="...")
        
        # Add node with attributes
        G.add_node(
            i,
            pos=(umap_embeddings[i, 0], umap_embeddings[i, 1]),
            cluster=labels[i],
            page=chunk_metadata[i]["page"],
            centrality=centrality_scores[i],
            text=short_text
        )
    
    # Calculate edge weights based on embedding similarity
    # Only create edges between similar items to avoid cluttered graph
    threshold = 0.7  # Cosine similarity threshold
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            # Only add edges within same cluster
            if labels[i] != labels[j]:
                continue
                
            # Calculate cosine similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            
            # Add edge if similarity is above threshold
            if sim > threshold:
                G.add_edge(i, j, weight=sim)
    
    # Create the plot
    plt.figure(figsize=(20, 16))
    
    # Get positions from UMAP
    pos = {i: (umap_embeddings[i, 0], umap_embeddings[i, 1]) for i in range(len(umap_embeddings))}
    
    # Draw nodes
    node_sizes = [50 + 200 * centrality_scores[i] for i in range(len(centrality_scores))]
    node_colors = [labels[i] for i in G.nodes()]
    
    # Create colormap
    num_clusters = len(set(labels))
    cmap = plt.cm.get_cmap('tab20', num_clusters)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=0.5,
        alpha=0.5,
        edge_color='gray'
    )
    
    # Add cluster labels at centroids
    for label, keywords in cluster_keywords.items():
        # Get all nodes in this cluster
        cluster_nodes = [n for n, d in G.nodes(data=True) if d['cluster'] == label]
        
        if not cluster_nodes:
            continue
            
        # Calculate centroid
        x_coords = [pos[n][0] for n in cluster_nodes]
        y_coords = [pos[n][1] for n in cluster_nodes]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        
        # Get top keywords
        top_keywords = ", ".join([word for word, score in keywords[:3]])
        
        # Add text
        plt.text(
            centroid_x, centroid_y,
            f"Topic {label}:\n{top_keywords}",
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            ha='center', va='center'
        )
    
    # Set title and remove axis
    plt.title("CASAC Knowledge Network: Relationships Between Concepts", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {output_path}")
    plt.close()

def main():
    # Load text files
    texts, file_names = load_text_files()
    print(f"Loaded {len(texts)} text files")
    
    # Extract paragraphs with better boundary detection
    paragraphs, paragraph_metadata = extract_paragraphs(texts)
    print(f"Extracted {len(paragraphs)} paragraphs")
    
    # Create overlapping chunks
    chunks, chunk_metadata = create_overlapping_chunks(paragraphs, paragraph_metadata, window_size=2, overlap=1)
    print(f"Created {len(chunks)} overlapping chunks")
    
    # Create embeddings with stronger model
    embeddings = create_embeddings(chunks, model_name="all-mpnet-base-v2")
    
    # Apply UMAP with better parameters
    umap_embeddings = apply_umap_with_parameters(
        embeddings, 
        n_neighbors=20,  # More neighbors for more global structure
        min_dist=0.1,    # Balance between local and global
        metric='cosine'  # Better for text
    )
    
    # Apply hierarchical clustering
    labels = apply_hierarchical_clustering(embeddings, n_clusters=8)
    
    # Identify central concepts within clusters
    centrality_scores = identify_central_concepts(embeddings, labels)
    
    # Extract keywords for clusters
    cluster_keywords = extract_keywords(chunks, labels)
    
    # Create advanced visualization
    create_advanced_visualization(
        umap_embeddings, 
        chunk_metadata, 
        labels, 
        centrality_scores, 
        cluster_keywords,
        chunks
    )
    
    # Create network visualization
    create_networkx_visualization(
        embeddings,
        umap_embeddings,
        labels,
        cluster_keywords,
        chunk_metadata,
        chunks,
        centrality_scores
    )
    
    print("Done!")

if __name__ == "__main__":
    main()