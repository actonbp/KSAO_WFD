import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch
import matplotlib.gridspec as gridspec
from pathlib import Path
import matplotlib as mpl

# Create directory for visualization
Path("creative_viz").mkdir(exist_ok=True)

def create_data_visualization():
    """Create a clear visualization of the raw TIFF data and processing workflow."""
    # Set up the figure with a clean aesthetic
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(16, 12), dpi=100)
    fig.patch.set_facecolor('white')
    
    # Add title
    plt.suptitle('From TIFF Images to Knowledge: The CASAC Analysis Journey', 
                fontsize=24, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.94, 'Visualization of the raw data extraction and analysis process', 
                fontsize=16, ha='center', style='italic')
    
    # Create grid layout
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # ============== Section 1: TIFF Images ==============
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Source Data: 11 TIFF Images from CASAC Training Materials', fontsize=18)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Create representative TIFF images
    for i in range(5):  # Show 5 of the 11 images
        # Create a subfigure for each TIFF image
        x_pos = 0.1 + i * 0.17
        ax_tiff = fig.add_axes([x_pos, 0.77, 0.15, 0.12])
        
        # Generate a text-like pattern
        np.random.seed(i + 42)
        rows, cols = 30, 100
        density = np.zeros((rows, cols))
        
        # Create line patterns that look like text
        for j in range(rows):
            if j % 3 == 0:  # Skip some lines for spacing
                continue
            start = np.random.randint(0, 10)
            end = np.random.randint(cols - 30, cols)
            density[j, start:end] = np.random.uniform(0.5, 1.0, end-start)
        
        # Show the image with a blue colormap
        ax_tiff.imshow(density, cmap='Blues')
        ax_tiff.set_xticks([])
        ax_tiff.set_yticks([])
        ax_tiff.set_title(f'DOC00{i}.tif', fontsize=10)
        
        # Add a border
        for spine in ax_tiff.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('#666666')
    
    # Add a note about remaining files
    plt.figtext(0.5, 0.75, "... and 6 more TIFF files", 
                ha='center', va='center', fontsize=10, style='italic')
    
    # ============== Section 2: OCR Process ==============
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Step 1: OCR Text Extraction', fontsize=16)
    ax2.text(0.5, 0.85, 'Tesseract OCR', fontsize=14, ha='center', weight='bold')
    ax2.text(0.5, 0.6, 'Image → Text Conversion', fontsize=12, ha='center')
    ax2.text(0.5, 0.4, """
    • Optical Character Recognition
    • Image preprocessing
    • Text recognition
    • Character extraction
    • Raw text output
    """, fontsize=10, ha='center', va='center')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add a fancy box around the OCR information
    ax2.add_patch(Rectangle((0.05, 0.05), 0.9, 0.85, fill=True, 
                           alpha=0.2, facecolor='royalblue', 
                           edgecolor='blue', linewidth=2,
                           transform=ax2.transAxes))
    
    # ============== Section 3: Text Processing ==============
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Step 2: Text Processing', fontsize=16)
    
    # Create extracted text examples
    processed_text = """
    Sample extracted text:
    
    CHAPTER 1: SCIENTIFIC PERSPECTIVES
    ON SUBSTANCE USE DISORDERS AND RECOVERY
    
    substance use disorder (SUD) is uncontrolled use of a 
    substance despite harmful consequences
    
    neurotransmitters that enable neurons to communicate
    
    brain reward pathways adapt to repeated exposure
    """
    
    ax3.text(0.5, 0.5, processed_text, fontsize=10, ha='center', va='center', 
             family='monospace', linespacing=1.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add a fancy box around the text
    ax3.add_patch(Rectangle((0.05, 0.05), 0.9, 0.85, fill=True, 
                           alpha=0.2, facecolor='forestgreen', 
                           edgecolor='green', linewidth=2,
                           transform=ax3.transAxes))
    
    # ============== Section 4: NLP Processing ==============
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title('Step 3: NLP Analysis', fontsize=16)
    
    # Describe NLP processing
    nlp_text = """
    Advanced Processing:
    
    • Sentence Transformers embeddings
       (all-mpnet-base-v2 model)
    
    • UMAP dimensionality reduction
       (n_neighbors=30, min_dist=0.1)
    
    • K-Means clustering (8 clusters)
    
    • Domain classification
    
    • Topic extraction
    """
    
    ax4.text(0.5, 0.5, nlp_text, fontsize=10, ha='center', va='center',
             linespacing=1.5)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Add a fancy box around the NLP information
    ax4.add_patch(Rectangle((0.05, 0.05), 0.9, 0.85, fill=True, 
                           alpha=0.2, facecolor='darkviolet', 
                           edgecolor='purple', linewidth=2,
                           transform=ax4.transAxes))
    
    # ============== Section 5: Key Concepts ==============
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title('Key Concepts Discovered in the CASAC Materials', fontsize=18)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    
    # Define the key concepts and their relationships
    concept_info = [
        {"name": "Substance Use Terminology", "category": "Foundations", "level": 0, "pos": 0.2},
        {"name": "Brain Function & Structure", "category": "Foundations", "level": 0, "pos": 0.5},
        {"name": "Reward System", "category": "Foundations", "level": 0, "pos": 0.8},
        {"name": "Dopamine & Neurotransmitters", "category": "Foundations", "level": 1, "pos": 0.35},
        {"name": "Addiction as Disease", "category": "Foundations", "level": 1, "pos": 0.65},
        {"name": "Risk Factors", "category": "Assessment", "level": 2, "pos": 0.25},
        {"name": "Use Continuum", "category": "Assessment", "level": 2, "pos": 0.5},
        {"name": "Harmful Use", "category": "Assessment", "level": 2, "pos": 0.75},
        {"name": "Person-First Language", "category": "Practice", "level": 3, "pos": 0.5}
    ]
    
    # Category colors
    category_colors = {
        "Foundations": "#4361ee",
        "Assessment": "#4cc9f0",
        "Practice": "#7209b7"
    }
    
    # Draw the concept nodes
    for concept in concept_info:
        x = concept["pos"]
        y = 0.8 - concept["level"] * 0.2
        
        # Create a box for the concept
        box = FancyBboxPatch(
            (x - 0.12, y - 0.03),
            0.24, 0.06,
            boxstyle="round,pad=0.3,rounding_size=0.2",
            facecolor=category_colors[concept["category"]],
            alpha=0.8,
            transform=ax5.transAxes,
            zorder=2
        )
        ax5.add_patch(box)
        
        # Add the concept name
        ax5.text(x, y, concept["name"], ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    
    # Add connections between concepts
    connections = [
        # From level 0 to level 1
        (0.2, 0.8, 0.35, 0.6),  # Terminology to Dopamine
        (0.5, 0.8, 0.35, 0.6),  # Brain Function to Dopamine
        (0.5, 0.8, 0.65, 0.6),  # Brain Function to Addiction
        (0.8, 0.8, 0.65, 0.6),  # Reward System to Addiction
        
        # From level 1 to level 2
        (0.35, 0.6, 0.25, 0.4), # Dopamine to Risk Factors
        (0.35, 0.6, 0.5, 0.4),  # Dopamine to Use Continuum
        (0.65, 0.6, 0.5, 0.4),  # Addiction to Use Continuum
        (0.65, 0.6, 0.75, 0.4), # Addiction to Harmful Use
        
        # From level 2 to level 3
        (0.25, 0.4, 0.5, 0.2),  # Risk Factors to Person-First
        (0.5, 0.4, 0.5, 0.2),   # Use Continuum to Person-First
        (0.75, 0.4, 0.5, 0.2)   # Harmful Use to Person-First
    ]
    
    # Draw the connections
    for start_x, start_y, end_x, end_y in connections:
        con = ConnectionPatch(
            xyA=(start_x, start_y), xyB=(end_x, end_y),
            coordsA="axes fraction", coordsB="axes fraction",
            axesA=ax5, axesB=ax5,
            arrowstyle="-|>", linewidth=1.0,
            color='gray', alpha=0.6
        )
        ax5.add_artist(con)
    
    # Add a legend for categories
    legend_items = []
    for category, color in category_colors.items():
        legend_items.append(
            Rectangle((0, 0), 1, 1, color=color, alpha=0.8)
        )
    
    ax5.legend(
        legend_items,
        category_colors.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=10
    )
    
    # ============== Add connecting arrows between sections ==============
    # Arrow from Section 1 to Section 2
    con1 = ConnectionPatch(
        xyA=(0.2, 0.72), xyB=(0.2, 0.65),
        coordsA="figure fraction", coordsB="figure fraction",
        arrowstyle="-|>", linewidth=2.0,
        color='royalblue'
    )
    fig.add_artist(con1)
    
    # Arrow from Section 2 to Section 3
    con2 = ConnectionPatch(
        xyA=(0.33, 0.5), xyB=(0.4, 0.5),
        coordsA="figure fraction", coordsB="figure fraction",
        arrowstyle="-|>", linewidth=2.0,
        color='forestgreen'
    )
    fig.add_artist(con2)
    
    # Arrow from Section 3 to Section 4
    con3 = ConnectionPatch(
        xyA=(0.6, 0.5), xyB=(0.67, 0.5),
        coordsA="figure fraction", coordsB="figure fraction",
        arrowstyle="-|>", linewidth=2.0,
        color='darkviolet'
    )
    fig.add_artist(con3)
    
    # Arrow from Section 4 to Section 5
    con4 = ConnectionPatch(
        xyA=(0.8, 0.4), xyB=(0.8, 0.32),
        coordsA="figure fraction", coordsB="figure fraction",
        arrowstyle="-|>", linewidth=2.0,
        color='purple'
    )
    fig.add_artist(con4)
    
    # ============== Add explanatory footer ==============
    footer_text = """
    This visualization depicts the journey from raw TIFF images to a structured knowledge representation of CASAC training materials.
    The process begins with OCR extraction of text from 11 TIFF images, followed by text processing and advanced NLP techniques
    to discover key concepts and their relationships. This allows addiction counselor trainees to explore the material through
    interactive visualizations that highlight semantic connections between important topics.
    """
    
    plt.figtext(0.5, 0.02, footer_text, ha='center', fontsize=10, 
               bbox=dict(facecolor='whitesmoke', edgecolor='lightgray', 
                        boxstyle='round,pad=0.5', alpha=0.8))
    
    # Save the figure with high resolution
    plt.savefig("creative_viz/casac_data_journey.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive data journey visualization created at: creative_viz/casac_data_journey.png")

if __name__ == "__main__":
    create_data_visualization()