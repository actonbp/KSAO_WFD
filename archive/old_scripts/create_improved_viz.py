import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import random
import os
from pathlib import Path
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba

# Create directory for visualization
Path("creative_viz").mkdir(exist_ok=True)

def create_tiff_representation():
    """Create an improved creative visualization representing the TIFF images and raw text extraction."""
    # Set up the figure with a clean, modern aesthetic
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.patch.set_facecolor('#f8f9fa')
    
    # Create a grid layout
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 2, 0.7])
    
    # Title and subtitle
    fig.text(0.5, 0.97, 'From Raw TIFF Images to Knowledge Visualization', 
             fontsize=24, fontweight='bold', ha='center', color='#333333')
    fig.text(0.5, 0.93, 'Visual representation of the text extraction and analysis process for CASAC training materials', 
             fontsize=14, ha='center', color='#666666', style='italic')
    
    # ==== Section 1: TIFF Images ====
    ax_tiff = fig.add_subplot(gs[0, 0])
    ax_tiff.set_title('Source Data: 11 TIFF Images', fontsize=16, pad=10, fontweight='bold')
    ax_tiff.axis('off')
    
    # Create a grid of TIFF image representations
    tiff_colors = ['#4361ee', '#3a0ca3', '#7209b7', '#560bad', '#480ca8', '#3f37c9']
    rows, cols = 2, 3
    for i in range(6):  # Show 6 representative images
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[0, 0], aspect='equal')
        
        # Position the mini-plots
        left = 0.05 + col * 0.3
        bottom = 0.75 - row * 0.4
        width, height = 0.25, 0.35
        ax.set_position([left, bottom, width, height])
        
        # Create a pixelated representation of a TIFF file
        np.random.seed(i + 42)
        text_density = np.random.rand(30, 100)
        text_density[text_density < 0.7] = 0  # Create "text-like" lines
        
        color = tiff_colors[i % len(tiff_colors)]
        ax.imshow(text_density, cmap='Blues', alpha=0.7, aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'DOC00{i}.tif', fontsize=10, color=color)
        
        # Add a fancy border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor(color)
    
    # Add a note about all 11 files
    fig.text(0.17, 0.51, "Note: Only 6 of 11 images shown", 
             fontsize=10, style='italic', ha='center')
    
    # ==== Section 2: OCR Process ====
    ax_ocr = fig.add_subplot(gs[0, 1])
    ax_ocr.set_title('OCR Text Extraction', fontsize=16, pad=10, fontweight='bold')
    ax_ocr.axis('off')
    
    # Create OCR process icon (simple text-based representation)
    ax_ocr_img = fig.add_axes([0.4, 0.55, 0.2, 0.28])
    ax_ocr_img.text(0.5, 0.6, "OCR", fontsize=40, ha='center', va='center',
                   fontweight='bold', color='#3f37c9')
    ax_ocr_img.text(0.5, 0.3, "Tesseract", fontsize=20, ha='center', va='center',
                   fontweight='bold', color='#7209b7')
    # Add a border
    for spine in ax_ocr_img.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('#4361ee')
    ax_ocr_img.set_xlim(0, 1)
    ax_ocr_img.set_ylim(0, 1)
    ax_ocr_img.axis('off')
    
    # Add OCR explanation
    ocr_text = """
    Tesseract OCR Process:
    
    1. Image preprocessing
    2. Text recognition
    3. Character extraction
    4. Word formation
    5. Raw text output
    """
    fig.text(0.5, 0.7, ocr_text, fontsize=12, ha='center', 
             bbox=dict(facecolor='#e9ecef', alpha=0.7, boxstyle='round,pad=0.7', 
                      edgecolor='#adb5bd'))
    
    # Add arrows from TIFF to OCR
    arrow_props = dict(arrowstyle='->', linewidth=2, color='#6c757d')
    fig.patches.append(plt.arrow(0.3, 0.7, 0.05, 0, transform=fig.transFigure, 
                               **arrow_props))
    
    # ==== Section 3: Raw Text Output ====
    ax_text = fig.add_subplot(gs[0, 2])
    ax_text.set_title('Extracted Raw Text', fontsize=16, pad=10, fontweight='bold')
    ax_text.axis('off')
    
    # Display extracted text fragments
    text_fragments = [
        "CHAPTER 1: SCIENTIFIC PERSPECTIVES",
        "ON SUBSTANCE USE DISORDERS AND RECOVERY",
        "substance use disorder (SUD) is uncontrolled use",
        "dopamine surge in the reward circuit",
        "brain reward pathways adapt to repeated exposure",
        "neurotransmitters that enable communication",
        "harmful use has significant consequences",
        "continuum from no use to severe substance use disorder"
    ]
    
    # Create a text box with faux OCR output
    text_box = fig.add_axes([0.7, 0.55, 0.25, 0.28])
    text_box.axis('off')
    text_box.set_facecolor('#f8f9fa')
    for spine in text_box.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('#adb5bd')
    
    for i, text in enumerate(text_fragments[:6]):
        y_pos = 0.8 - i * 0.13
        text_box.text(0.05, y_pos, text, fontsize=9, color='#212529')
    
    # Add arrow from OCR to text
    fig.patches.append(plt.arrow(0.63, 0.7, 0.05, 0, transform=fig.transFigure,
                               **arrow_props))
    
    # ==== Section 4: Data Processing Pipeline ====
    ax_process = fig.add_subplot(gs[1, :])
    ax_process.set_title('Data Processing Pipeline', fontsize=18, pad=15, fontweight='bold')
    ax_process.axis('off')
    
    # Create the processing pipeline visualization
    pipeline_steps = [
        {"name": "Raw Text", "desc": "Unformatted text from OCR", "color": "#4cc9f0"},
        {"name": "Text Cleaning", "desc": "Remove noise, normalize", "color": "#4895ef"},
        {"name": "Embedding Generation", "desc": "SentenceTransformers\nall-mpnet-base-v2", "color": "#4361ee"},
        {"name": "Dimensionality Reduction", "desc": "UMAP preserves semantic relationships", "color": "#3f37c9"},
        {"name": "Topic Clustering", "desc": "Identify related concepts", "color": "#3a0ca3"},
        {"name": "Knowledge Visualization", "desc": "Interactive semantic maps", "color": "#7209b7"}
    ]
    
    # Draw the processing flow
    box_width = 0.13
    box_height = 0.15
    for i, step in enumerate(pipeline_steps):
        # Position each step box
        x_pos = 0.03 + i * (box_width + 0.03)
        y_pos = 0.35
        
        # Create fancy box for each step
        box = FancyBboxPatch(
            (x_pos, y_pos), box_width, box_height,
            boxstyle="round,pad=0.6,rounding_size=0.2",
            facecolor=step["color"],
            alpha=0.85,
            transform=fig.transFigure, 
            zorder=2
        )
        fig.patches.append(box)
        
        # Add step title
        fig.text(x_pos + box_width/2, y_pos + box_height*0.75, step["name"],
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='white', zorder=3)
        
        # Add step description
        fig.text(x_pos + box_width/2, y_pos + box_height*0.35, step["desc"],
               ha='center', va='center', fontsize=9,
               color='white', zorder=3, linespacing=1.3)
        
        # Add connecting arrow
        if i < len(pipeline_steps) - 1:
            arrow_x = x_pos + box_width
            arrow_y = y_pos + box_height/2
            arrow_dx = 0.03
            arrow_dy = 0
            fig.patches.append(plt.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy,
                                       transform=fig.transFigure,
                                       color='white', width=0.001, head_width=0.01,
                                       length_includes_head=True, zorder=3))
    
    # ==== Section 5: Example Data Elements ====
    ax_examples = fig.add_subplot(gs[2, :])
    ax_examples.set_title('Key Concepts Extracted from TIFF Images', fontsize=16, pad=10, fontweight='bold')
    ax_examples.axis('off')
    
    # Create concept bubbles - these are the key concepts found in the text
    concepts = [
        {"text": "Substance Use Disorders", "size": 0.07, "color": "#4cc9f0"},
        {"text": "Brain Reward Circuit", "size": 0.06, "color": "#4895ef"},
        {"text": "Dopamine Release", "size": 0.05, "color": "#4361ee"},
        {"text": "Neurotransmitters", "size": 0.055, "color": "#3f37c9"},
        {"text": "Addiction as Disease", "size": 0.065, "color": "#3a0ca3"},
        {"text": "Risk Factors", "size": 0.05, "color": "#7209b7"},
        {"text": "Continuum of Use", "size": 0.06, "color": "#560bad"},
        {"text": "Person-First Language", "size": 0.05, "color": "#b5179e"}
    ]
    
    # Draw concept bubbles
    for i, concept in enumerate(concepts):
        # Arrange in a circular pattern
        angle = i * (2 * np.pi / len(concepts))
        radius = 0.18
        x = 0.5 + radius * np.cos(angle)
        y = 0.13 + radius * np.sin(angle)
        
        # Create bubble
        bubble = plt.Circle((x, y), concept["size"], 
                          color=concept["color"], alpha=0.8, transform=fig.transFigure)
        fig.patches.append(bubble)
        
        # Add text
        fig.text(x, y, concept["text"], ha='center', va='center', 
               fontsize=10, color='white', fontweight='bold',
               transform=fig.transFigure)
    
    # Add explanatory caption
    caption = """
    This visualization illustrates the process of extracting knowledge from 11 TIFF images from the CASAC training materials.
    Starting with scanned pages, OCR technology extracts raw text which is then processed through a natural language
    processing pipeline to identify key concepts, topics, and their relationships - ultimately creating interactive
    visualizations that help counselors understand the complex material.
    """
    
    fig.text(0.5, 0.02, caption, ha='center', va='center', fontsize=10, 
           color='#495057', style='italic',
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5', 
                    edgecolor='#dee2e6'))
    
    # Save the figure with high resolution
    plt.savefig("creative_viz/enhanced_data_journey.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Enhanced creative visualization created at: creative_viz/enhanced_data_journey.png")

if __name__ == "__main__":
    create_tiff_representation()