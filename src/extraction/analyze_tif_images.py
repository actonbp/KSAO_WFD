import os
import pytesseract
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from pathlib import Path
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns

# Create directories if they don't exist
def create_directories():
    Path("text_output").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)

# Extract text from images using Tesseract OCR
def extract_text_from_images(image_paths, output_folder="text_output"):
    all_text = ""
    page_texts = []
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image: {img_path}")
        img = Image.open(img_path)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(img)
        
        # Save text to file
        text_file = os.path.join(output_folder, f"page_{i+1}.txt")
        with open(text_file, "w") as f:
            f.write(text)
        
        all_text += text + "\n\n"
        page_texts.append(text)
        print(f"Extracted text saved to: {text_file}")
    
    # Save the combined text
    with open(os.path.join(output_folder, "full_text.txt"), "w") as f:
        f.write(all_text)
    
    return all_text, page_texts

# Clean the extracted text
def clean_text(text):
    # Remove special characters, but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,;?!-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Generate embeddings using TF-IDF
def generate_embeddings(texts):
    # Clean texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2
    )
    embeddings = vectorizer.fit_transform(cleaned_texts)
    
    # Get feature names for later analysis
    feature_names = vectorizer.get_feature_names_out()
    
    return embeddings, feature_names, vectorizer

# Visualize embeddings using PCA and t-SNE
def visualize_embeddings(embeddings, page_numbers, output_folder="visualizations"):
    # Convert sparse matrix to dense
    dense_embeddings = embeddings.toarray()
    
    # PCA for dimensionality reduction (intermediate step)
    # Use n_components that's appropriate for our dataset size
    n_components = min(5, min(dense_embeddings.shape[0], dense_embeddings.shape[1]) - 1)
    if dense_embeddings.shape[1] > n_components and n_components > 0:
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(dense_embeddings)
    else:
        reduced_embeddings = dense_embeddings
    
    # t-SNE for 2D visualization
    tsne = TSNE(n_components=2, perplexity=min(5, len(reduced_embeddings)-1), 
                random_state=42, n_iter=2000)
    tsne_results = tsne.fit_transform(reduced_embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create color map
    colors = cm.rainbow(np.linspace(0, 1, len(tsne_results)))
    
    # Plot points
    for i, (x, y) in enumerate(tsne_results):
        plt.scatter(x, y, c=[colors[i]], s=100, alpha=0.8)
        plt.annotate(f"Page {page_numbers[i]}", (x, y), fontsize=9)
    
    plt.title("t-SNE visualization of page embeddings")
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(output_folder, "page_embeddings.png"), dpi=300)
    plt.close()
    
    print(f"Visualization saved to: {os.path.join(output_folder, 'page_embeddings.png')}")
    
    return tsne_results

# Analyze the top terms for each page
def analyze_top_terms(vectorizer, embeddings, page_numbers, n_top_terms=10):
    feature_names = vectorizer.get_feature_names_out()
    dense_embeddings = embeddings.toarray()
    
    top_terms = {}
    for i, page in enumerate(page_numbers):
        # Get the TF-IDF scores for this page
        tfidf_scores = [(feature_names[j], dense_embeddings[i][j]) 
                        for j in range(len(feature_names))]
        
        # Sort by score and get top terms
        sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_terms[page] = sorted_terms[:n_top_terms]
    
    return top_terms

# Generate word clouds for all pages and individual pages
def generate_wordclouds(page_texts, all_text, output_folder="visualizations"):
    # Create a word cloud for the entire document
    wordcloud = WordCloud(width=800, height=400, background_color="white", 
                         max_words=100, contour_width=3, contour_color='steelblue').generate(all_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud - All Pages")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "all_pages_wordcloud.png"), dpi=300)
    plt.close()
    
    # Create word clouds for individual pages
    for i, text in enumerate(page_texts):
        if len(text.strip()) > 50:  # Only create wordcloud if there's enough text
            wordcloud = WordCloud(width=800, height=400, background_color="white", 
                                 max_words=50, contour_width=3, contour_color='steelblue').generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Word Cloud - Page {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"page_{i+1}_wordcloud.png"), dpi=300)
            plt.close()
    
    print(f"Word clouds saved to: {output_folder}")

# Create term frequency heatmap
def create_term_frequency_heatmap(page_texts, top_n=20, output_folder="visualizations"):
    # Create a count vectorizer to get term frequencies
    cv = CountVectorizer(max_features=100, stop_words='english', min_df=2)
    counts = cv.fit_transform(page_texts).toarray()
    
    # Get the most common terms across all documents
    total_counts = np.sum(counts, axis=0)
    top_indices = np.argsort(total_counts)[-top_n:]
    top_terms = [cv.get_feature_names_out()[i] for i in top_indices]
    
    # Create a dataframe with the top terms for each page
    df = pd.DataFrame(counts[:, top_indices], columns=top_terms)
    df.index = [f"Page {i+1}" for i in range(len(page_texts))]
    
    # Create a heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5)
    plt.title(f"Top {top_n} Terms Frequency Across Pages")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "term_frequency_heatmap.png"), dpi=300)
    plt.close()
    
    print(f"Term frequency heatmap saved to: {output_folder}/term_frequency_heatmap.png")

# Main function
def main():
    # Create necessary directories
    create_directories()
    
    # Get all TIF images in the images folder
    image_folder = "images"
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith('.tif')])
    
    if not image_paths:
        print("No TIF images found in the 'images' folder.")
        return
    
    print(f"Found {len(image_paths)} TIF images")
    
    # Extract text from images
    all_text, page_texts = extract_text_from_images(image_paths)
    
    # Generate embeddings
    page_numbers = [i+1 for i in range(len(page_texts))]
    embeddings, feature_names, vectorizer = generate_embeddings(page_texts)
    
    # Visualize embeddings
    tsne_results = visualize_embeddings(embeddings, page_numbers)
    
    # Analyze top terms per page
    top_terms = analyze_top_terms(vectorizer, embeddings, page_numbers)
    
    # Print top terms for each page
    print("\nTop terms for each page:")
    for page, terms in top_terms.items():
        print(f"\nPage {page}:")
        for term, score in terms:
            print(f"  - {term}: {score:.4f}")
    
    # Generate word clouds
    generate_wordclouds(page_texts, all_text)
    
    # Create term frequency heatmap
    create_term_frequency_heatmap(page_texts)

if __name__ == "__main__":
    main()