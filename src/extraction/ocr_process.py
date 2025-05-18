import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from pathlib import Path

# Create directories if they don't exist
def create_directories():
    Path("images").mkdir(exist_ok=True)
    Path("text_output").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)

# Convert PDF to images
def pdf_to_images(pdf_path, output_folder="images"):
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
        print(f"Saved image: {img_path}")
    
    return image_paths

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
    
    # PCA for dimensionality reduction to 50 dimensions (intermediate step)
    if dense_embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
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

# Main function
def main():
    # Create necessary directories
    create_directories()
    
    # Path to the PDF file
    pdf_path = "CASAC 350 HR Program.pdf"
    
    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path)
    
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

if __name__ == "__main__":
    main()