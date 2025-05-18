#!/bin/bash

# This script tests if the KSAO pipeline is correctly configured
# and ready to process new documents

echo "===== Testing KSAO Processing Pipeline ====="
echo "This test will verify that the pipeline components are working correctly."
echo "It uses a minimal subset of existing data to test without making API calls."
echo "For a full test with API calls, add the --test-file parameter with a TIFF filename."
echo 

# Run the pipeline test script
python3 src/utils/test_pipeline.py "$@"

# Check the exit code
if [ $? -eq 0 ]; then
    echo 
    echo "All pipeline components are functioning correctly!"
    echo "You can now process new documents following the guide in docs/NEW_DOCUMENT_PROCESSING.md"
else
    echo 
    echo "Pipeline test failed. Please check the output for details."
    echo "Fix any reported issues before processing new documents."
fi