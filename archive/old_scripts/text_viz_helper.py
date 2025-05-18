import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.cm as cm
import re
from pathlib import Path

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

def clean_text(text):
    """Clean text for analysis."""
    # Remove special characters but keep spaces and punctuation
    text = re.sub(r'[^\w\s.,;?!-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_tfidf_embeddings(texts, max_features=1000, min_df=2):
    """Generate TF-IDF embeddings."""
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=min_df
    )
    embeddings = vectorizer.fit_transform(cleaned_texts)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return embeddings, feature_names, vectorizer

def visualize_embeddings_2d(embeddings, labels, output_path="visualizations/embeddings_2d.png", 
                         method="tsne", title="Document Embeddings"):
    """Visualize embeddings in 2D."""
    # Convert sparse matrix to dense if needed
    if hasattr(embeddings, "toarray"):
        dense_embeddings = embeddings.toarray()
    else:
        dense_embeddings = embeddings
    
    # Dimensionality reduction
    if method == "tsne":
        # PCA first if high-dimensional
        if dense_embeddings.shape[1] > 50:
            n_components = min(50, min(dense_embeddings.shape) - 1)
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(dense_embeddings)
        else:
            reduced = dense_embeddings
            
        # t-SNE for final 2D
        perplexity = min(30, len(reduced) - 1)
        if perplexity < 5:
            perplexity = 2
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        result_2d = tsne.fit_transform(reduced)
    else:  # Use PCA or SVD directly
        if dense_embeddings.shape[0] < dense_embeddings.shape[1]:
            # SVD for sparse matrices with more features than samples
            svd = TruncatedSVD(n_components=2, random_state=42)
            result_2d = svd.fit_transform(dense_embeddings)
        else:
            # PCA otherwise
            pca = PCA(n_components=2, random_state=42)
            result_2d = pca.fit_transform(dense_embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create color map
    colors = cm.rainbow(np.linspace(0, 1, len(result_2d)))
    
    # Plot points
    for i, (x, y) in enumerate(result_2d):
        plt.scatter(x, y, c=[colors[i]], s=100, alpha=0.8)
        plt.annotate(labels[i], (x, y), fontsize=9)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return result_2d

def create_wordcloud(text, output_path="visualizations/wordcloud.png", 
                   title="Word Cloud", max_words=100):
    """Create a word cloud from text."""
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        max_words=max_words, 
        contour_width=3, 
        contour_color='steelblue'
    ).generate(text)
    
    # Display word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_term_frequency_heatmap(texts, top_n=20, output_path="visualizations/term_heatmap.png",
                               labels=None, title="Term Frequency Across Documents"):
    """Create a heatmap of term frequencies."""
    # Create a count vectorizer
    cv = CountVectorizer(max_features=100, stop_words='english', min_df=2)
    counts = cv.fit_transform(texts).toarray()
    
    # Get the most common terms
    total_counts = np.sum(counts, axis=0)
    top_indices = np.argsort(total_counts)[-top_n:]
    top_terms = [cv.get_feature_names_out()[i] for i in top_indices]
    
    # Create a dataframe
    if labels is None:
        labels = [f"Document {i+1}" for i in range(len(texts))]
    
    df = pd.DataFrame(counts[:, top_indices], columns=top_terms)
    df.index = labels
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return df

def analyze_top_terms(vectorizer, embeddings, document_labels, n_top_terms=10):
    """Analyze top terms by TF-IDF score."""
    feature_names = vectorizer.get_feature_names_out()
    dense_embeddings = embeddings.toarray()
    
    top_terms = {}
    for i, doc in enumerate(document_labels):
        # Get the TF-IDF scores
        tfidf_scores = [(feature_names[j], dense_embeddings[i][j]) 
                      for j in range(len(feature_names))]
        
        # Sort by score and get top terms
        sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_terms[doc] = sorted_terms[:n_top_terms]
    
    return top_terms

def print_top_terms(top_terms):
    """Print top terms in a readable format."""
    for doc, terms in top_terms.items():
        print(f"\n{doc}:")
        for term, score in terms:
            print(f"  - {term}: {score:.4f}")

def batch_process_visualizations(texts, file_names, output_dir="visualizations"):
    """Process all visualizations in batch."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Combine all texts
    all_text = " ".join(cleaned_texts)
    
    # Create labels
    labels = [f"Page {i+1}" for i in range(len(texts))]
    
    # Generate embeddings
    embeddings, feature_names, vectorizer = generate_tfidf_embeddings(cleaned_texts)
    
    # Create visualizations
    create_wordcloud(all_text, output_path=f"{output_dir}/all_pages_wordcloud.png", 
                   title="Word Cloud - All Pages")
    
    visualize_embeddings_2d(embeddings, labels, 
                          output_path=f"{output_dir}/embeddings_2d.png",
                          title="Document Embeddings")
    
    create_term_frequency_heatmap(cleaned_texts, labels=labels,
                               output_path=f"{output_dir}/term_heatmap.png")
    
    # Individual wordclouds
    for i, text in enumerate(cleaned_texts):
        if len(text.strip()) > 50:  # Only create if enough text
            create_wordcloud(text, 
                          output_path=f"{output_dir}/page_{i+1}_wordcloud.png",
                          title=f"Word Cloud - Page {i+1}")
    
    # Analyze top terms
    top_terms = analyze_top_terms(vectorizer, embeddings, labels)
    
    return {
        "embeddings": embeddings,
        "vectorizer": vectorizer,
        "top_terms": top_terms
    }

if __name__ == "__main__":
    # Example usage
    texts, file_names = load_text_files()
    results = batch_process_visualizations(texts, file_names)
    print_top_terms(results["top_terms"])