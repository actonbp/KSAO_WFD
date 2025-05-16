import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
import textwrap
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import umap
from sklearn.metrics import silhouette_score

# Create directories
Path("optimal_viz").mkdir(exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# CASAC knowledge domains (mapped to certification requirements)
CASAC_DOMAINS = {
    "Knowledge": "Understanding of substance use disorders and addiction science",
    "Assessment": "Screening, evaluation, and diagnosis",
    "Treatment": "Counseling approaches and modalities",
    "Professional": "Ethics, documentation, and professional responsibilities",
    "Support": "Recovery maintenance and continuing care"
}

# Keywords associated with each domain - used for automatic categorization
DOMAIN_KEYWORDS = {
    "Knowledge": ["brain", "neurotransmitters", "substance", "addiction", "disorder", "dopamine", 
                 "reward", "disease", "neurons", "model", "scientific", "circuit", "function"],
    "Assessment": ["screening", "evaluation", "diagnosis", "assessment", "symptoms", "criteria",
                   "measuring", "testing", "stages", "problematic", "harmful", "continuum"],
    "Treatment": ["treatment", "counseling", "therapy", "intervention", "modalities", "approaches",
                 "behavioral", "cognitive", "therapeutic", "motivational", "recovery"],
    "Professional": ["ethics", "documentation", "records", "confidentiality", "law", "legal",
                    "professional", "standards", "boundaries", "conduct", "supervision"],
    "Support": ["support", "community", "resources", "family", "continuing", "maintenance",
               "relapse", "prevention", "aftercare", "group", "self-help", "peer"]
}

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

def assign_domains(paragraphs):
    """Assign CASAC knowledge domains to paragraphs based on keyword presence."""
    domains = []
    
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        
        # Count keyword matches for each domain
        domain_scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            domain_scores[domain] = sum(1 for keyword in keywords if keyword.lower() in paragraph_lower)
        
        # Assign the domain with the highest score, or "Knowledge" if tied/no matches
        if sum(domain_scores.values()) == 0:
            domains.append("Knowledge")  # Default domain
        else:
            max_score = max(domain_scores.values())
            best_domains = [domain for domain, score in domain_scores.items() if score == max_score]
            domains.append(best_domains[0])  # Take the first if tied
    
    return domains

def create_embeddings(paragraphs, model_name="all-mpnet-base-v2"):
    """Create embeddings using a stronger Sentence Transformers model."""
    print(f"Creating embeddings with model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings with progress bar
    embeddings = model.encode(paragraphs, show_progress_bar=True)
    
    return embeddings

def determine_optimal_clusters(embeddings, max_clusters=15):
    """Determine the optimal number of clusters using KMeans and silhouette score."""
    print("Determining optimal number of clusters...")
    
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        # Use KMeans for easier silhouette scoring
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(set(labels)) > 1:  # Ensure at least 2 clusters for silhouette score
            score = silhouette_score(embeddings, labels, metric='cosine')
            silhouette_scores.append((n_clusters, score))
            print(f"  Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
    # Find the best number of clusters
    best_n_clusters, best_score = max(silhouette_scores, key=lambda x: x[1])
    print(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.4f})")
    
    return best_n_clusters

def apply_hierarchical_clustering(embeddings, n_clusters):
    """Apply hierarchical clustering to the embeddings."""
    print(f"Applying hierarchical clustering with {n_clusters} clusters...")
    
    # Create similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Apply hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
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

def extract_keywords(paragraphs, labels):
    """Extract keywords that characterize each cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Group paragraphs by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(paragraphs[i])
    
    # Extract keywords for each cluster
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        max_df=0.7,
        min_df=2
    )
    
    # Fit on all paragraphs
    vectorizer.fit(paragraphs)
    
    # Extract top keywords for each cluster
    cluster_keywords = {}
    for label, cluster_paragraphs in clusters.items():
        # Combine all text in this cluster
        cluster_text = " ".join(cluster_paragraphs)
        
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

def assign_cluster_titles(cluster_keywords):
    """Assign descriptive titles to clusters based on keywords."""
    cluster_titles = {}
    
    for cluster, keywords in cluster_keywords.items():
        # Get top 4 keywords
        top_words = [word for word, _ in keywords[:4]]
        
        # Map these to a descriptive title based on domain knowledge
        if any(word in ["substance", "addiction", "disorder", "use"] for word in top_words):
            title = "Substance Use Terminology & Concepts"
        elif any(word in ["brain", "neuron", "neurotransmitter", "dopamine"] for word in top_words):
            title = "Neurobiology of Addiction"
        elif any(word in ["reward", "circuit", "pleasure", "surge"] for word in top_words):
            title = "Brain Reward Pathways"
        elif any(word in ["term", "language", "negative", "stigma"] for word in top_words):
            title = "Person-First Language & Terminology"
        elif any(word in ["risk", "factor", "trauma", "childhood"] for word in top_words):
            title = "Risk Factors & Development"
        elif any(word in ["stage", "continuum", "harmful", "problem"] for word in top_words):
            title = "Substance Use Continuum & Stages"
        elif any(word in ["treatment", "recovery", "therapy", "counseling"] for word in top_words):
            title = "Treatment & Recovery Approaches"
        else:
            # Default to using the keywords
            title = f"Topic: {', '.join(top_words[:3])}"
        
        cluster_titles[cluster] = title
    
    return cluster_titles

def create_domain_visualization(umap_embeddings, paragraph_metadata, labels, domains, 
                             cluster_titles, paragraphs,
                             output_path="optimal_viz/domain_visualization.html"):
    """Create an interactive visualization organized by CASAC domains."""
    print("Creating domain-based visualization...")
    
    # Domain color mapping
    domain_colors = {
        "Knowledge": "#4C72B0",  # Blue
        "Assessment": "#55A868",  # Green
        "Treatment": "#C44E52",   # Red
        "Professional": "#8172B3", # Purple
        "Support": "#CCB974"      # Yellow
    }
    
    # Prepare data for visualization
    df = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "page": [meta["page"] for meta in paragraph_metadata],
        "cluster": labels,
        "domain": domains,
        "domain_color": [domain_colors[domain] for domain in domains],
        "cluster_title": [cluster_titles[label] for label in labels],
        "text": [textwrap.shorten(paragraph, width=300, placeholder="...") for paragraph in paragraphs]
    })
    
    # Create the visualization
    fig = px.scatter(
        df, 
        x="x", 
        y="y", 
        color="domain",
        color_discrete_map=domain_colors,
        hover_data=["page", "cluster_title", "text"],
        title="CASAC Knowledge Domains: Scientific Perspectives on Substance Use Disorders",
        height=800,
        width=1000,
        opacity=0.8,
        symbol="cluster_title",
        # Use same symbol within cluster but different colors by domain
        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down']
    )
    
    # Add cluster labels at centroids
    for label, title in cluster_titles.items():
        cluster_df = df[df["cluster"] == label]
        centroid_x = cluster_df["x"].mean()
        centroid_y = cluster_df["y"].mean()
        
        # Add annotation for cluster
        fig.add_annotation(
            x=centroid_x,
            y=centroid_y,
            text=title,
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
    
    # Add domain descriptions
    domain_descriptions = []
    for domain, description in CASAC_DOMAINS.items():
        # Count how many paragraphs in this domain
        count = sum(1 for d in domains if d == domain)
        domain_descriptions.append(f"<b>{domain}</b> ({count} paragraphs): {description}")
    
    domain_text = "<br>".join(domain_descriptions)
    
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=domain_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=4,
        align="left",
        font=dict(size=10)
    )
    
    # Improve the hover template
    fig.update_traces(
        hovertemplate="<b>Page %{customdata[0]}</b><br>Topic: %{customdata[1]}<br>Domain: %{marker.color}<br><br>%{customdata[2]}<extra></extra>"
    )
    
    # Layout improvements
    fig.update_layout(
        legend_title="CASAC Knowledge Domains",
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

def main():
    # Load text files
    texts, file_names = load_text_files()
    print(f"Loaded {len(texts)} text files")
    
    # Extract paragraphs with better boundary detection
    paragraphs, paragraph_metadata = extract_paragraphs(texts)
    print(f"Extracted {len(paragraphs)} paragraphs")
    
    # Assign CASAC domains to paragraphs
    domains = assign_domains(paragraphs)
    domain_counts = {}
    for domain in domains:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    print(f"Domain distribution: {domain_counts}")
    
    # Create embeddings with stronger model
    embeddings = create_embeddings(paragraphs, model_name="all-mpnet-base-v2")
    
    # Determine optimal number of clusters
    n_clusters = determine_optimal_clusters(embeddings, max_clusters=10)
    
    # Apply hierarchical clustering with optimal number of clusters
    labels = apply_hierarchical_clustering(embeddings, n_clusters)
    
    # Apply UMAP with better parameters
    umap_embeddings = apply_umap_with_parameters(
        embeddings, 
        n_neighbors=20,  # More neighbors for more global structure
        min_dist=0.1,    # Balance between local and global
        metric='cosine'  # Better for text
    )
    
    # Extract keywords for clusters
    cluster_keywords = extract_keywords(paragraphs, labels)
    
    # Assign meaningful titles to clusters
    cluster_titles = assign_cluster_titles(cluster_keywords)
    
    # Create domain-based visualization
    create_domain_visualization(
        umap_embeddings, 
        paragraph_metadata, 
        labels, 
        domains, 
        cluster_titles,
        paragraphs
    )
    
    print("Done!")

if __name__ == "__main__":
    main()