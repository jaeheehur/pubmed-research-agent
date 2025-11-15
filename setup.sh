#!/bin/bash

# Setup script for PubMed Research Agent

echo "=================================================="
echo "Setting up PubMed Research Agent Environment"
echo "=================================================="

# Create conda environment
echo "Creating conda environment: pubmed_py312"
conda create -n pubmed_py312 python=3.12 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pubmed_py312

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy models for biomedical NER
echo "Downloading spaCy biomedical models..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz

# Create Jupyter kernel
echo "Creating Jupyter kernel..."
python -m ipykernel install --user --name=pubmed_py312 --display-name="Python 3.12 (PubMed Agent)"

echo "=================================================="
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  conda activate pubmed_py312"
echo "=================================================="
