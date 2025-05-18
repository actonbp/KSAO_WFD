# Scan Directory

This directory contains scanned materials from the CASAC certification program.

## Contents

- **Appendices.tif**: Appendices from the CASAC materials
- **Chapter *.tif**: Scanned chapters from the CASAC textbook
- **PP202.tif**: Additional materials

## Using These Scans

These scans are raw input data for the OCR process. To extract text from them:

```bash
python ../../src/main.py --ocr
```

The extracted text will be saved to the `../text_output/` directory.
EOL < /dev/null