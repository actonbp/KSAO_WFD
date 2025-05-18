#!/usr/bin/env python3
import os
import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure the Gemini API client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the client
genai_client = genai.Client(api_key=api_key)

def create_output_directory(output_dir="output/network_visualizations"):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    return output_dir

def extract_ksao_relationships(analysis_file):
    """Use Gemini to extract KSAO relationships from the textbook analysis."""
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_text = f.read()
    except Exception as e:
        print(f"Error reading analysis file: {e}")
        return None
    
    model = genai_client.models.get_model("gemini-2.5-pro-preview-05-06")
    
    prompt = f"""
    I have an analysis of KSAOs (Knowledge, Skills, Abilities, and Other characteristics) for substance use disorder counselors.
    Please extract structured data about the KSAOs and their relationships to create a network graph.
    
    For each KSAO identified in the text, extract:
    1. KSAO ID (assign a numeric identifier)
    2. Name
    3. Classification (Knowledge, Skill, Ability, or Other)
    4. Level (general or specialized)
    5. Parent KSAOs (IDs of KSAOs that this one is a subset of or depends on)
    
    Return the data in this JSON format:
    {{
        "nodes": [
            {{
                "id": 1,
                "name": "Example KSAO Name",
                "classification": "Knowledge|Skill|Ability|Other",
                "level": "General|Specialized"
            }},
            ...
        ],
        "links": [
            {{
                "source": 1,
                "target": 2,
                "relationship": "prerequisite|dimension|foundation"
            }},
            ...
        ]
    }}
    
    Here is the analysis text:
    {analysis_text[:80000]}  # Increased limit for Gemini 2.5 Pro's larger context window
    """
    
    try:
        print("Extracting KSAO relationships...")
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": 0.1,  # Low temperature for precise data extraction
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192
            },
            safety_settings={
                "harassment": "block_none",
                "hate_speech": "block_none",
                "sexually_explicit": "block_none",
                "dangerous_content": "block_none"
            }
        )
        response_text = response.text
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            network_data = json.loads(json_str)
            return network_data
        else:
            print("Could not extract valid JSON from the response")
            return None
            
    except Exception as e:
        print(f"Error extracting relationships: {e}")
        return None

def create_network_graph(network_data, output_dir):
    """Create and visualize a network graph from the KSAO relationships."""
    if not network_data:
        return
    
    G = nx.DiGraph()
    
    # Add nodes
    node_colors = {'Knowledge': 'skyblue', 'Skill': 'lightgreen', 
                  'Ability': 'salmon', 'Other': 'lightgray'}
    
    node_colors_map = []
    for node in network_data.get('nodes', []):
        G.add_node(node['id'], 
                  name=node['name'],
                  classification=node.get('classification', 'Unknown'),
                  level=node.get('level', 'Unknown'))
        
        color = node_colors.get(node.get('classification', 'Unknown'), 'yellow')
        node_colors_map.append(color)
    
    # Add edges
    for link in network_data.get('links', []):
        G.add_edge(link['source'], link['target'], 
                   relationship=link.get('relationship', 'unknown'))
    
    # Save network data
    output_file = os.path.join(output_dir, "ksao_network_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(network_data, f, indent=2)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Create standard network graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color=node_colors_map, node_size=300, alpha=0.8)
    
    # Add labels with smaller font
    labels = {node: G.nodes[node]['name'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
    
    plt.title("KSAO Network for SUD Counselors")
    plt.savefig(os.path.join(output_dir, "ksao_network_graph.png"), dpi=300, bbox_inches='tight')
    
    # Create hierarchical graph (if possible)
    try:
        plt.figure(figsize=(14, 10))
        pos_hierarchy = nx.nx_agraph.graphviz_layout(G, prog="dot")
        nx.draw(G, pos_hierarchy, with_labels=False, node_color=node_colors_map, node_size=300, alpha=0.8)
        nx.draw_networkx_labels(G, pos_hierarchy, labels, font_size=8, font_family='sans-serif')
        
        plt.title("Hierarchical KSAO Network for SUD Counselors")
        plt.savefig(os.path.join(output_dir, "ksao_hierarchical_graph.png"), dpi=300, bbox_inches='tight')
    except:
        print("Could not create hierarchical layout (pygraphviz may not be installed)")
    
    # Create classification-based subgraphs
    for classification, color in node_colors.items():
        subgraph_nodes = [n for n in G.nodes if G.nodes[n]['classification'] == classification]
        if subgraph_nodes:
            subgraph = G.subgraph(subgraph_nodes)
            plt.figure(figsize=(10, 8))
            pos_sub = nx.spring_layout(subgraph, seed=42)
            nx.draw(subgraph, pos_sub, with_labels=False, node_color=color, node_size=300, alpha=0.8)
            
            # Add labels
            sub_labels = {node: G.nodes[node]['name'] for node in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos_sub, sub_labels, font_size=8, font_family='sans-serif')
            
            plt.title(f"{classification} Network for SUD Counselors")
            plt.savefig(os.path.join(output_dir, f"ksao_{classification.lower()}_graph.png"), dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")
    return G

def merge_similar_competencies(all_competencies, similarity_threshold=0.7):
    """Merge similar competencies to avoid duplication."""
    model = genai_client.models.get_model("gemini-2.5-pro-preview-05-06")
    
    prompt = f"""
    I have a list of competencies and skills extracted from a text. 
    Please merge any that are very similar or redundant, and return a consolidated list.
    
    Use a similarity threshold of {similarity_threshold} (where 1.0 means identical).
    
    For each merged competency, use the name, description, and other attributes from the most comprehensive entry, 
    and include a list of the merged items.
    
    Original competencies:
    {json.dumps(all_competencies, indent=2)}
    
    Return a JSON array of the consolidated competencies with the same structure as the input, 
    but with an additional field "merged_from" for any entries that merged others.
    """
    
    try:
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192
            },
            safety_settings={
                "harassment": "block_none",
                "hate_speech": "block_none",
                "sexually_explicit": "block_none",
                "dangerous_content": "block_none"
            }
        )
        response_text = response.text
        
        # Try to find JSON structure in the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            merged_competencies = json.loads(json_str)
            return merged_competencies
        else:
            print("Could not find valid JSON in the merge response")
            return all_competencies
    except Exception as e:
        print(f"Error merging competencies: {e}")
        return all_competencies

def main():
    parser = argparse.ArgumentParser(description="Visualize KSAO relationships as network graphs")
    parser.add_argument("--analysis-file", default="output/full_analysis/textbook_ksao_analysis.txt", 
                        help="File containing the textbook analysis")
    parser.add_argument("--output-dir", default="output/network_visualizations", 
                        help="Directory for visualization output")
    
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    
    # Extract KSAO relationships
    network_data = extract_ksao_relationships(args.analysis_file)
    
    if network_data:
        # Create network graph
        create_network_graph(network_data, output_dir)

if __name__ == "__main__":
    main() 