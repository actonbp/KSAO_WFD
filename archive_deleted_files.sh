#!/bin/bash
# Script to archive files showing as deleted in git status

echo "=== Archiving files shown as deleted in git status ==="

# Create archive directories if they don't exist
mkdir -p archive/old_scripts
mkdir -p archive/old_directories/academic_viz
mkdir -p archive/old_directories/creative_viz 
mkdir -p archive/old_directories/final_viz
mkdir -p archive/old_directories/images
mkdir -p archive/old_directories/interactive_viz
mkdir -p archive/old_directories/optimal_viz
mkdir -p archive/old_directories/text_output
mkdir -p archive/old_directories/visualizations

# Move Python scripts to old_scripts
echo "Moving Python scripts to archive/old_scripts..."
git checkout -- academic_visualization.py && mv academic_visualization.py archive/old_scripts/
git checkout -- analyze_tif_images.py && mv analyze_tif_images.py archive/old_scripts/
git checkout -- create_improved_viz.py && mv create_improved_viz.py archive/old_scripts/
git checkout -- create_raw_data_viz.py && mv create_raw_data_viz.py archive/old_scripts/
git checkout -- domain_enhanced_viz.py && mv domain_enhanced_viz.py archive/old_scripts/
git checkout -- export_embeddings.py && mv export_embeddings.py archive/old_scripts/
git checkout -- final_data_viz.py && mv final_data_viz.py archive/old_scripts/
git checkout -- final_visualization.py && mv final_visualization.py archive/old_scripts/
git checkout -- ocr_process.py && mv ocr_process.py archive/old_scripts/
git checkout -- optimal_visualization.py && mv optimal_visualization.py archive/old_scripts/
git checkout -- run_analysis.py && mv run_analysis.py archive/old_scripts/
git checkout -- text_viz_helper.py && mv text_viz_helper.py archive/old_scripts/
git checkout -- umap_study_guide.py && mv umap_study_guide.py archive/old_scripts/

# Move visualization files and directories
echo "Moving visualization files to archive/old_directories..."
git checkout -- academic_viz/academic_casac_map.png && mv academic_viz/academic_casac_map.png archive/old_directories/academic_viz/
git checkout -- creative_viz/casac_data_journey.png && mv creative_viz/casac_data_journey.png archive/old_directories/creative_viz/
git checkout -- creative_viz/raw_data_representation.png && mv creative_viz/raw_data_representation.png archive/old_directories/creative_viz/

# Move final viz files
echo "Moving final_viz files to archive/old_directories..."
git checkout -- final_viz/casac_learning_map.html && mv final_viz/casac_learning_map.html archive/old_directories/final_viz/
git checkout -- final_viz/casac_learning_map.png && mv final_viz/casac_learning_map.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_0.png && mv final_viz/wordcloud_cluster_0.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_1.png && mv final_viz/wordcloud_cluster_1.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_2.png && mv final_viz/wordcloud_cluster_2.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_3.png && mv final_viz/wordcloud_cluster_3.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_4.png && mv final_viz/wordcloud_cluster_4.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_5.png && mv final_viz/wordcloud_cluster_5.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_6.png && mv final_viz/wordcloud_cluster_6.png archive/old_directories/final_viz/
git checkout -- final_viz/wordcloud_cluster_7.png && mv final_viz/wordcloud_cluster_7.png archive/old_directories/final_viz/

# Move image files
echo "Moving image files to archive/old_directories..."
git checkout -- images/DOC000.tif && mv images/DOC000.tif archive/old_directories/images/
git checkout -- images/DOC001.tif && mv images/DOC001.tif archive/old_directories/images/
git checkout -- images/DOC002.tif && mv images/DOC002.tif archive/old_directories/images/
git checkout -- images/DOC003.tif && mv images/DOC003.tif archive/old_directories/images/
git checkout -- images/DOC004.tif && mv images/DOC004.tif archive/old_directories/images/
git checkout -- images/DOC005.tif && mv images/DOC005.tif archive/old_directories/images/
git checkout -- images/DOC006.tif && mv images/DOC006.tif archive/old_directories/images/
git checkout -- images/DOC007.tif && mv images/DOC007.tif archive/old_directories/images/
git checkout -- images/DOC008.tif && mv images/DOC008.tif archive/old_directories/images/
git checkout -- images/DOC009.tif && mv images/DOC009.tif archive/old_directories/images/
git checkout -- images/DOC010.tif && mv images/DOC010.tif archive/old_directories/images/

# Move interactive viz
echo "Moving interactive_viz files to archive/old_directories..."
git checkout -- interactive_viz/umap_interactive.html && mv interactive_viz/umap_interactive.html archive/old_directories/interactive_viz/

# Move optimal viz
echo "Moving optimal_viz files to archive/old_directories..."
git checkout -- optimal_viz/advanced_clustering.html && mv optimal_viz/advanced_clustering.html archive/old_directories/optimal_viz/
git checkout -- optimal_viz/advanced_clustering.png && mv optimal_viz/advanced_clustering.png archive/old_directories/optimal_viz/
git checkout -- optimal_viz/domain_visualization.html && mv optimal_viz/domain_visualization.html archive/old_directories/optimal_viz/
git checkout -- optimal_viz/domain_visualization.png && mv optimal_viz/domain_visualization.png archive/old_directories/optimal_viz/
git checkout -- optimal_viz/knowledge_graph.png && mv optimal_viz/knowledge_graph.png archive/old_directories/optimal_viz/

# Move text_output files
echo "Moving text_output files to archive/old_directories..."
git checkout -- text_output/full_text.txt && mv text_output/full_text.txt archive/old_directories/text_output/
git checkout -- text_output/page_1.txt && mv text_output/page_1.txt archive/old_directories/text_output/
git checkout -- text_output/page_10.txt && mv text_output/page_10.txt archive/old_directories/text_output/
git checkout -- text_output/page_11.txt && mv text_output/page_11.txt archive/old_directories/text_output/
git checkout -- text_output/page_2.txt && mv text_output/page_2.txt archive/old_directories/text_output/
git checkout -- text_output/page_3.txt && mv text_output/page_3.txt archive/old_directories/text_output/
git checkout -- text_output/page_4.txt && mv text_output/page_4.txt archive/old_directories/text_output/
git checkout -- text_output/page_5.txt && mv text_output/page_5.txt archive/old_directories/text_output/
git checkout -- text_output/page_6.txt && mv text_output/page_6.txt archive/old_directories/text_output/
git checkout -- text_output/page_7.txt && mv text_output/page_7.txt archive/old_directories/text_output/
git checkout -- text_output/page_8.txt && mv text_output/page_8.txt archive/old_directories/text_output/
git checkout -- text_output/page_9.txt && mv text_output/page_9.txt archive/old_directories/text_output/

# Move visualization files
echo "Moving visualization files to archive/old_directories..."
git checkout -- visualizations/all_pages_wordcloud.png && mv visualizations/all_pages_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_10_wordcloud.png && mv visualizations/page_10_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_11_wordcloud.png && mv visualizations/page_11_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_1_wordcloud.png && mv visualizations/page_1_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_2_wordcloud.png && mv visualizations/page_2_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_3_wordcloud.png && mv visualizations/page_3_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_4_wordcloud.png && mv visualizations/page_4_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_5_wordcloud.png && mv visualizations/page_5_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_6_wordcloud.png && mv visualizations/page_6_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_7_wordcloud.png && mv visualizations/page_7_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_8_wordcloud.png && mv visualizations/page_8_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_9_wordcloud.png && mv visualizations/page_9_wordcloud.png archive/old_directories/visualizations/
git checkout -- visualizations/page_embeddings.png && mv visualizations/page_embeddings.png archive/old_directories/visualizations/
git checkout -- visualizations/term_frequency_heatmap.png && mv visualizations/term_frequency_heatmap.png archive/old_directories/visualizations/
git checkout -- visualizations/umap_clusters.png && mv visualizations/umap_clusters.png archive/old_directories/visualizations/

echo "=== Archiving complete! ==="
echo "All files have been moved to the archive directory structure."