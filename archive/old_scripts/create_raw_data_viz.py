import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import random
import os
from pathlib import Path
import matplotlib.patheffects as path_effects

# Create directory for visualization
Path("creative_viz").mkdir(exist_ok=True)

def create_tiff_representation():
    """Create a creative visualization representing the TIFF images and raw text extraction."""
    fig = plt.figure(figsize=(16, 10), facecolor='#f0f0f0')
    
    # Custom background
    plt.rcParams['axes.facecolor'] = '#f0f0f0'
    
    # Create a custom colormap for the TIFF representation
    colors = [(0.2, 0.2, 0.5), (0.4, 0.4, 0.8), (0.7, 0.7, 1.0)]
    cmap_name = 'blue_gradient'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Main title
    plt.suptitle('From Raw Images to Knowledge Discovery', fontsize=24, fontweight='bold', y=0.95)
    plt.figtext(0.5, 0.91, 'Visualizing the journey from TIFF images to text data for CASAC training materials', 
                ha='center', fontsize=14, style='italic')
    
    # Create a grid layout
    gs = gridspec.GridSpec(10, 16, figure=fig)
    
    # Create the TIFF images representation (left side)
    ax1 = fig.add_subplot(gs[:5, :7])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('TIFF Images (Source Data)', fontsize=14, pad=10)
    
    # Draw representations of TIFF files
    tiff_frames = []
    tiff_labels = []
    for i in range(11):
        x = 0.1 + 0.08 * (i % 3)
        y = 0.8 - 0.2 * (i // 3)
        width, height = 0.18, 0.15
        
        if i == 10:  # Last one centered at bottom
            x = 0.38
            y = 0.05
        
        # Create a TIFF representation with random noise pattern
        np.random.seed(i)
        noise = np.random.rand(20, 20)
        tiff_ax = fig.add_axes([x, y, width, height])
        tiff_ax.imshow(noise, cmap=cm, interpolation='nearest')
        tiff_ax.set_xticks([])
        tiff_ax.set_yticks([])
        tiff_ax.set_title(f'DOC00{i}.tif', fontsize=8)
        
        # Add a fancy border
        for spine in tiff_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('navy')
        
        tiff_frames.append(tiff_ax)
        tiff_labels.append(f'DOC00{i}.tif')
    
    # Create the OCR process visualization (middle)
    ax2 = fig.add_subplot(gs[:5, 7:10])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    ax2.set_title('OCR Process', fontsize=14, pad=10)
    
    # Draw arrows and OCR process
    for i, tiff_ax in enumerate(tiff_frames):
        # Calculate positions for arrows
        src_x = tiff_ax.get_position().x1
        src_y = tiff_ax.get_position().y0 + tiff_ax.get_position().height/2
        
        dst_x = 0.55  # Destination is the OCR box
        dst_y = 0.5 - 0.04 * i  # Stagger the arrivals
        
        # Create arrow connection
        arrow = plt.annotate('', 
                          xy=(dst_x, dst_y), xycoords='figure fraction',
                          xytext=(src_x, src_y), textcoords='figure fraction',
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                         color='navy', alpha=0.6, linewidth=1.5))
    
    # Add OCR box
    ocr_box = plt.annotate('Tesseract OCR\nText Extraction', 
                         xy=(0.55, 0.5), xycoords='figure fraction',
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                                  edgecolor='navy', linewidth=2))
    
    # Create the extracted text visualization (right side)
    ax3 = fig.add_subplot(gs[:5, 10:])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('Extracted Raw Text', fontsize=14, pad=10)
    
    # Sample text content from the CASAC materials
    text_fragments = [
        "substance use disorders",
        "neurotransmitters and neural pathways",
        "brain reward circuit",
        "dopamine release during pleasure",
        "addiction as a chronic brain disease",
        "genetic and environmental risk factors",
        "problematic substance use patterns",
        "CASAC certification requirements",
        "neurobiology of addiction",
        "tolerance and withdrawal symptoms",
        "evidence-based treatment approaches",
        "stigmatizing vs. person-first language",
        "stages of substance use continuum",
        "harmful use and its consequences",
        "scientific perspectives on substance use"
    ]
    
    # Create text flow from OCR to raw text
    text_y_positions = []
    for i, text in enumerate(text_fragments):
        y_pos = 0.8 - i * 0.05
        text_y_positions.append(y_pos)
        
        # Add text with random horizontal position
        x_jitter = random.uniform(0.65, 0.9)
        text_obj = plt.figtext(x_jitter, y_pos, text, fontsize=9, 
                              ha='left', va='center',
                              bbox=dict(facecolor='white', edgecolor='gray', 
                                      boxstyle='round,pad=0.3', alpha=0.8))
        
    # Draw connection from OCR to text
    for y_pos in text_y_positions:
        arrow = plt.annotate('', 
                          xy=(0.65, y_pos), xycoords='figure fraction',
                          xytext=(0.58, 0.5), textcoords='figure fraction',
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                         color='navy', alpha=0.4, linewidth=1))
    
    # Create the data processing flow (bottom)
    ax4 = fig.add_subplot(gs[5:, :])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title('From Raw Text to Knowledge Discovery', fontsize=16, pad=15)
    
    # Create a processing pipeline visualization
    steps = [
        "Raw Text", 
        "Text Cleaning", 
        "Embeddings\n(SentenceTransformers)", 
        "Dimensionality\nReduction (UMAP)", 
        "Clustering", 
        "Knowledge Map"
    ]
    
    # Draw the processing pipeline
    box_positions = []
    for i, step in enumerate(steps):
        x = 0.1 + i * 0.16
        y = 0.22
        width, height = 0.12, 0.08
        
        box = FancyBboxPatch((x, y), width, height, 
                            boxstyle=f"round,pad=0.3,rounding_size=0.4",
                            facecolor=plt.cm.Blues(0.5 + i * 0.1),
                            edgecolor='navy', linewidth=2,
                            transform=fig.transFigure, zorder=2)
        fig.patches.append(box)
        box_positions.append((x, y, width, height))
        
        # Add step name
        step_text = plt.figtext(x + width/2, y + height/2, step, 
                              ha='center', va='center', fontsize=10,
                              fontweight='bold', color='white', zorder=3)
        
        # Add path effect for better visibility
        step_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='navy')])
    
    # Connect the steps with arrows
    for i in range(len(box_positions) - 1):
        src_x = box_positions[i][0] + box_positions[i][2]
        src_y = box_positions[i][1] + box_positions[i][3]/2
        
        dst_x = box_positions[i+1][0]
        dst_y = box_positions[i+1][1] + box_positions[i+1][3]/2
        
        arrow = plt.annotate('',
                          xy=(dst_x, dst_y), xycoords='figure fraction',
                          xytext=(src_x, src_y), textcoords='figure fraction',
                          arrowprops=dict(arrowstyle='-|>', color='navy', 
                                         linewidth=2, mutation_scale=15),
                          zorder=1)
    
    # Add explanation text
    explanation_text = """
    This visualization represents the data journey:
    
    1. Source Data: 11 TIFF images from CASAC training materials
    2. OCR Process: Extracting text from images using Tesseract
    3. Raw Text: Unprocessed text fragments containing domain knowledge
    4. Processing Pipeline: Transforming raw text into structured knowledge
    
    The original TIFF images contain textbook content on substance use disorders,
    addiction neurobiology, treatment approaches, and counselor certification.
    """
    
    plt.figtext(0.5, 0.05, explanation_text, ha='center', va='center', 
               fontsize=9, bbox=dict(facecolor='white', alpha=0.8, 
                                     boxstyle='round,pad=0.5'))
    
    # Save the figure
    plt.savefig("creative_viz/raw_data_representation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Creative visualization of raw data created at: creative_viz/raw_data_representation.png")

if __name__ == "__main__":
    create_tiff_representation()