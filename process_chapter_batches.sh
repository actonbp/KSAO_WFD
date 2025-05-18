#!/bin/bash

# This script processes OCR on TIFF files in batches to avoid timeouts
# Usage: ./process_chapter_batches.sh <chapter_filename> <batch_size>
# Example: ./process_chapter_batches.sh "Chapter 3.tif" 5

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <chapter_filename> [batch_size]"
    echo "Example: $0 \"Chapter 3.tif\" 5"
    exit 1
fi

CHAPTER_FILE="$1"
BATCH_SIZE="${2:-5}"  # Default batch size of 5 pages if not provided

# Check if the file exists
if [ ! -f "Scan/$CHAPTER_FILE" ]; then
    echo "Error: File Scan/$CHAPTER_FILE does not exist"
    exit 1
fi

# Get the number of pages in the TIFF file
# You need to install imagemagick for this: brew install imagemagick
NUM_PAGES=$(identify -format "%n\n" "Scan/$CHAPTER_FILE" | head -1)
echo "$CHAPTER_FILE has $NUM_PAGES pages"

# Clean filename for output
CLEAN_NAME=$(echo "$CHAPTER_FILE" | sed 's/.tif$//;s/ /_/g')

# Calculate number of batches
NUM_BATCHES=$(( (NUM_PAGES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Will process in $NUM_BATCHES batches of $BATCH_SIZE pages each"

# Process each batch
for (( i=0; i<NUM_BATCHES; i++ )); do
    START_PAGE=$(( i * BATCH_SIZE + 1 ))
    END_PAGE=$(( (i + 1) * BATCH_SIZE ))
    
    # Ensure last batch doesn't exceed total pages
    if [ $END_PAGE -gt $NUM_PAGES ]; then
        END_PAGE=$NUM_PAGES
    fi
    
    echo "====================================="
    echo "Processing batch $((i+1))/$NUM_BATCHES: Pages $START_PAGE to $END_PAGE"
    echo "====================================="
    
    python3 gemini_ocr.py --single-chapter "$CHAPTER_FILE" \
                         --input-dir Scan \
                         --output-dir "data/gemini_text_output" \
                         --start-page $START_PAGE \
                         --end-page $END_PAGE
    
    # Add a short pause between batches to avoid API rate limits
    echo "Pausing for 5 seconds before next batch..."
    sleep 5
done

echo "====================================="
echo "Finished processing $CHAPTER_FILE"
echo "====================================="

# Create or append to the full chapter file
OUTPUT_DIR="data/gemini_text_output/${CLEAN_NAME}"
FULL_FILE="data/gemini_text_output/${CLEAN_NAME}_full.txt"

# Check if there are page files
PAGE_FILES=$(ls "$OUTPUT_DIR"/page_*.txt 2>/dev/null)
if [ -z "$PAGE_FILES" ]; then
    echo "Error: No page files found in $OUTPUT_DIR"
    exit 1
fi

# Combine all page files into the full file
echo "Creating combined file for $CLEAN_NAME..."
cat "$OUTPUT_DIR"/page_*.txt > "$FULL_FILE"

echo "All batches have been processed. Full text saved to:"
echo "$FULL_FILE"