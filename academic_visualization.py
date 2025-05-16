import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
import textwrap
import matplotlib.cm as cm
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

# Create directories
Path("academic_viz").mkdir(exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# CASAC knowledge domains with more specific sub-categories
CASAC_DOMAINS = {
    "Foundations": {
        "description": "Scientific understanding of addiction",
        "subdomains": [
            "Brain structure and function",
            "Neurotransmitters and reward pathways",
            "Addiction as a disease",
            "Scientific terminology"
        ]
    },
    "Assessment": {
        "description": "Screening and diagnosis",
        "subdomains": [
            "Substance use continuum",
            "Risk factors",
            "Screening tools",
            "Diagnostic criteria"
        ]
    },
    "Practice": {
        "description": "Counseling approaches",
        "subdomains": [
            "Treatment modalities",
            "Communication skills",
            "Recovery support",
            "Person-first approach"
        ]
    }
}

# Keywords for domain classification - expanded for better detection
DOMAIN_KEYWORDS = {
    "Foundations": [
        # Brain/neuroscience terms
        "brain", "neuron", "neurotransmitter", "dopamine", "circuit", "reward",
        "function", "structure", "system", "pathway", "nucleus", "nerve",
        # Disease model terms
        "disease", "chronic", "relapsing", "biological", "genetic", "hereditary",
        # Scientific terminology
        "scientific", "research", "study", "evidence", "data", "theory", "model"
    ],
    
    "Assessment": [
        # Continuum of use
        "continuum", "stages", "progression", "harmful", "problematic", "risky",
        # Risk factors
        "risk", "factor", "vulnerability", "predisposition", "trauma", "adverse", 
        "childhood", "development", "environmental", "social",
        # Screening/diagnosis
        "assessment", "screening", "diagnosis", "criteria", "evaluation", 
        "symptom", "indicator", "measure", "test", "scale"
    ],
    
    "Practice": [
        # Treatment approaches
        "treatment", "therapy", "intervention", "counseling", "approach", "modality",
        "technique", "method", "practice", "strategy", "behavioral", "cognitive",
        # Person-first
        "person", "individual", "human", "client", "stigma", "language",
        "communication", "terminology", "dignity", "respect", "first",
        # Recovery
        "recovery", "support", "maintenance", "relapse", "prevention", 
        "continuing", "care", "aftercare", "community", "family"
    ]
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

def extract_meaningful_chunks(texts, chunk_size=5):
    """Extract meaningful chunks with paragraph and section awareness."""
    all_chunks = []
    chunk_metadata = []
    
    for doc_idx, text in enumerate(texts):
        page_num = doc_idx + 1
        
        # Split on double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Check if we have headers (all caps lines or numbered sections)
        headers = []
        header_indices = []
        
        for i, para in enumerate(paragraphs):
            first_line = para.split('\n')[0].strip()
            if (first_line.isupper() or 
                re.match(r'^[0-9]+\.', first_line) or 
                len(first_line) < 50 and first_line.endswith(':')):
                headers.append(first_line)
                header_indices.append(i)
        
        # Group paragraphs by section if headers found
        if headers and len(headers) > 1:
            sections = []
            
            # Create sections based on headers
            for i in range(len(header_indices)):
                start_idx = header_indices[i]
                end_idx = header_indices[i+1] if i+1 < len(header_indices) else len(paragraphs)
                section_text = "\n\n".join(paragraphs[start_idx:end_idx])
                sections.append(section_text)
            
            # Process each section
            for section_idx, section in enumerate(sections):
                # Split section into sentences
                try:
                    sentences = sent_tokenize(section)
                except Exception:
                    sentences = simple_sentence_tokenize(section)
                
                # Group sentences into chunks
                for i in range(0, len(sentences), chunk_size):
                    if i + chunk_size <= len(sentences):
                        chunk = " ".join(sentences[i:i+chunk_size])
                    else:
                        chunk = " ".join(sentences[i:])
                    
                    if len(chunk.strip()) < 50:  # Skip very small chunks
                        continue
                    
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        "doc_idx": doc_idx,
                        "page": page_num,
                        "section": section_idx,
                        "header": headers[section_idx] if section_idx < len(headers) else "",
                        "chunk_type": "section",
                        "chunk_text": chunk
                    })
        else:
            # If no clear sections, process paragraph by paragraph
            for para_idx, para in enumerate(paragraphs):
                if len(para.strip()) < 50:  # Skip very small paragraphs
                    continue
                
                all_chunks.append(para)
                chunk_metadata.append({
                    "doc_idx": doc_idx,
                    "page": page_num,
                    "paragraph": para_idx,
                    "chunk_type": "paragraph",
                    "chunk_text": para
                })
    
    return all_chunks, chunk_metadata

def assign_domain(text):
    """Assign the most likely CASAC domain based on keyword matching."""
    text_lower = text.lower()
    
    # Count keywords for each domain
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw.lower() in text_lower)
    
    # Return the domain with the highest score, or the first if tied
    max_score = max(scores.values())
    if max_score == 0:
        return "Foundations"  # Default domain
    
    for domain, score in scores.items():
        if score == max_score:
            return domain
    
    return "Foundations"  # Fallback

def create_embeddings(chunks, model_name="all-mpnet-base-v2"):
    """Create embeddings using a stronger Sentence Transformers model."""
    print(f"Creating embeddings with model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings with progress bar
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    return embeddings

def apply_cluster_analysis(embeddings, n_clusters=8):
    """Apply KMeans clustering to the embeddings."""
    print(f"Applying KMeans clustering with {n_clusters} clusters...")
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels

def apply_umap_with_parameters(embeddings, n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine'):
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

def extract_cluster_keywords(chunks, labels):
    """Extract keywords that characterize each cluster."""
    # Group chunks by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[i])
    
    # Extract keywords for each cluster
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2)  # Include bigrams for better cluster descriptions
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
        
        # Sort by score and take top keywords
        indices = np.argsort(scores)[::-1][:15]
        top_keywords = [(feature_names[i], scores[i]) for i in indices]
        
        cluster_keywords[label] = top_keywords
    
    return cluster_keywords

def assign_cluster_names(cluster_keywords):
    """Assign meaningful names to clusters based on keywords."""
    # Define meaningful names based on CASAC topics
    potential_names = {
        # Foundations domain
        "brain_function": ["brain", "neuron", "function", "structure", "complex"],
        "reward_system": ["reward", "dopamine", "pleasure", "circuit", "surge"],
        "addiction_science": ["disease", "addiction", "chronic", "model", "theory"],
        "terminology": ["term", "language", "substance", "definition", "concepts"],
        
        # Assessment domain
        "continuum": ["continuum", "stage", "progression", "harmful", "problematic"],
        "risk_factors": ["risk", "factor", "trauma", "childhood", "influence"],
        "screening": ["assessment", "screening", "evaluation", "criteria", "measure"],
        
        # Practice domain
        "treatment": ["treatment", "therapy", "counseling", "approach", "recovery"],
        "communication": ["person", "first", "language", "stigma", "communication"],
        "support": ["recovery", "support", "maintenance", "relapse", "prevention"]
    }
    
    cluster_names = {}
    
    for label, keywords in cluster_keywords.items():
        keyword_terms = [term for term, _ in keywords]
        
        # Score each potential name based on keyword matches
        name_scores = {}
        for name, terms in potential_names.items():
            score = sum(1 for term in terms if any(term in kw for kw in keyword_terms))
            name_scores[name] = score
        
        # Find the name with the highest score
        best_name = max(name_scores.items(), key=lambda x: x[1])
        
        if best_name[1] > 0:
            # Convert snake_case to Title Case and assign
            pretty_name = " ".join(word.capitalize() for word in best_name[0].split("_"))
            cluster_names[label] = pretty_name
        else:
            # Fallback to using top keywords
            top_terms = [term for term, _ in keywords[:3]]
            cluster_names[label] = f"Cluster {label}: {', '.join(top_terms)}"
    
    return cluster_names

def create_academic_visualization(umap_embeddings, chunks, chunk_metadata, labels, 
                                cluster_names, domains, cluster_keywords,
                                output_path="academic_viz/academic_casac_map.png"):
    """Create a comprehensive academic visualization with extensive text annotations."""
    print("Creating academic visualization...")
    
    # Prepare data
    df = pd.DataFrame({
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "page": [meta["page"] for meta in chunk_metadata],
        "cluster": labels,
        "cluster_name": [cluster_names[label] for label in labels],
        "domain": domains
    })
    
    # Domain color mapping
    domain_colors = {
        "Foundations": "#4C72B0",  # Blue
        "Assessment": "#55A868",   # Green
        "Practice": "#C44E52"      # Red
    }
    
    # Count chunks by domain and cluster
    domain_counts = df['domain'].value_counts().to_dict()
    cluster_counts = {}
    for label in np.unique(labels):
        cluster_counts[label] = len(df[df['cluster'] == label])
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Create colorful scatter plot
    for domain, color in domain_colors.items():
        domain_df = df[df['domain'] == domain]
        
        for cluster in domain_df['cluster'].unique():
            cluster_df = domain_df[domain_df['cluster'] == cluster]
            plt.scatter(
                cluster_df['x'], 
                cluster_df['y'],
                s=80,  # Larger points for better visibility
                c=color,
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5,
                label=f"{domain}: {cluster_names[cluster]}"
            )
    
    # Calculate and add cluster centroids and annotations
    for label in np.unique(labels):
        cluster_df = df[df['cluster'] == label]
        centroid_x = cluster_df['x'].mean()
        centroid_y = cluster_df['y'].mean()
        
        # Get domain distribution for this cluster
        cluster_domains = cluster_df['domain'].value_counts().to_dict()
        primary_domain = max(cluster_domains.items(), key=lambda x: x[1])[0] if cluster_domains else "Foundations"
        
        # Get top keywords
        top_keywords = ", ".join([word for word, _ in cluster_keywords[label][:5]])
        
        # Add annotation with box
        plt.annotate(
            f"{cluster_names[label]}\n({cluster_counts[label]} chunks)\n\nTop terms: {top_keywords}",
            xy=(centroid_x, centroid_y),
            xytext=(centroid_x + 0.5, centroid_y + 0.5),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=domain_colors[primary_domain],
                alpha=0.9
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
                color=domain_colors[primary_domain]
            ),
            fontsize=9,
            ha='center'
        )
    
    # Add title and subtitle
    plt.suptitle("CASAC Knowledge Map: Scientific Perspectives on Substance Use Disorders", 
                fontsize=20, y=0.98)
    
    plt.figtext(0.5, 0.94, 
               "Visualization of text embeddings from 11 pages of CASAC training materials using Sentence Transformers and UMAP",
               ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Add domain legend
    domain_patches = []
    for domain, color in domain_colors.items():
        count = domain_counts.get(domain, 0)
        domain_patches.append(
            mpatches.Patch(
                color=color,
                label=f"{domain} ({count} chunks): {CASAC_DOMAINS[domain]['description']}"
            )
        )
    
    # Add methodology explanation text box
    plt.figtext(0.5, 0.04, 
               "METHODOLOGY\n"
               "1. Text Extraction: Chunked text into meaningful semantic units\n"
               "2. Embeddings: Created vector representations using SentenceTransformers (all-mpnet-base-v2)\n"
               "3. Clustering: Applied KMeans to identify 8 topic clusters\n"
               "4. Dimensionality Reduction: UMAP projection preserves semantic relationships\n"
               "5. Domain Classification: Assigned knowledge domains based on keyword analysis",
               ha='center', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add visualization guide
    plt.figtext(0.05, 0.04,
               "HOW TO READ THIS VISUALIZATION\n"
               "• Each point represents a text chunk from the training material\n"
               "• Color indicates knowledge domain (blue=Foundations, green=Assessment, red=Practice)\n"
               "• Proximity indicates semantic similarity between concepts\n"
               "• Clusters show related topics that should be studied together\n"
               "• Text boxes display topic names and key terminology",
               fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add domain legend to upper right
    plt.figtext(0.85, 0.9,
              "KNOWLEDGE DOMAINS",
              fontsize=12, weight='bold',
              bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add domain descriptions
    for i, (domain, details) in enumerate(CASAC_DOMAINS.items()):
        color = domain_colors[domain]
        count = domain_counts.get(domain, 0)
        plt.figtext(0.85, 0.86 - i*0.04,
                  f"● {domain} ({count} chunks)",
                  color=color, fontsize=10, weight='bold')
        plt.figtext(0.85, 0.83 - i*0.04,
                  f"   {details['description']}",
                  color='black', fontsize=9)
        # Add subdomains
        for j, subdomain in enumerate(details['subdomains'][:2]):  # Show only first 2 subdomains
            plt.figtext(0.85, 0.80 - i*0.04 - j*0.025,
                      f"   - {subdomain}",
                      color='black', fontsize=8)
    
    # Remove axes
    plt.axis('off')
    
    # Add a subtle grid to help with spatial orientation
    plt.grid(True, linestyle='--', alpha=0.2)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # Adjust layout to accommodate text
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Academic visualization saved to {output_path}")

def main():
    # Load text files
    texts, file_names = load_text_files()
    print(f"Loaded {len(texts)} text files")
    
    # Extract meaningful chunks
    chunks, chunk_metadata = extract_meaningful_chunks(texts, chunk_size=5)
    print(f"Extracted {len(chunks)} meaningful chunks")
    
    # Assign domains based on content
    domains = [assign_domain(chunk) for chunk in chunks]
    domain_counts = {}
    for domain in domains:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    print(f"Domain distribution: {domain_counts}")
    
    # Create embeddings
    embeddings = create_embeddings(chunks)
    
    # Apply UMAP
    umap_embeddings = apply_umap_with_parameters(
        embeddings,
        n_neighbors=30,      # More global structure
        min_dist=0.1,        # Good separation
        metric='cosine'      # Better for text
    )
    
    # Determine number of clusters
    n_clusters = 8  # A good balance for this dataset
    
    # Apply clustering
    labels = apply_cluster_analysis(embeddings, n_clusters)
    
    # Extract keywords for each cluster
    cluster_keywords = extract_cluster_keywords(chunks, labels)
    
    # Assign meaningful names to clusters
    cluster_names = assign_cluster_names(cluster_keywords)
    print("Cluster names:")
    for label, name in cluster_names.items():
        print(f"  Cluster {label}: {name}")
    
    # Create academic visualization with comprehensive annotations
    create_academic_visualization(
        umap_embeddings,
        chunks,
        chunk_metadata,
        labels,
        cluster_names,
        domains,
        cluster_keywords
    )
    
    print("Done!")

if __name__ == "__main__":
    main()