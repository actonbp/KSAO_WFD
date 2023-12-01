import os
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from collections import Counter
import re


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def clean_text(text):
    """Cleans the extracted text."""
    # Remove special characters and numbers
    return re.sub(r"[^A-Za-z\s]", "", text)


def get_word_frequencies(text):
    """Calculates word frequencies."""
    words = text.lower().split()
    return Counter(words)


def create_frequency_graph(word_freq):
    """Creates a word frequency graph."""
    words, frequencies = zip(*word_freq.most_common(20))
    plt.figure(figsize=(15, 10))
    plt.barh(words, frequencies, color="skyblue")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Top 20 Words in PDF Documents")
    plt.gca().invert_yaxis()
    plt.show()


# Main script
if __name__ == "__main__":
    docs_folder = "docs"
    all_text = ""

    # Process each PDF in the 'docs' folder
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            all_text += clean_text(text)

    # Get word frequencies and create a graph
    word_frequencies = get_word_frequencies(all_text)
    create_frequency_graph(word_frequencies)
