#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment
source .venv/bin/activate

# Run the Python script
python generate_data.py
python extract_fields.py
python evaluate.py
python generate_pdf.py

# Deactivate the virtual environment
deactivate