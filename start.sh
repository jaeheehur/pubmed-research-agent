#!/bin/bash

# PubMed Research Agent - Start Script
# This script sets up the environment and runs the Streamlit app

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=================================================="
echo "PubMed Research Agent"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment
CONDA_ENV="pubmed_py312"
echo -e "${BLUE}Activating conda environment: ${CONDA_ENV}${NC}"

# Try different conda paths
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    source $(conda info --base)/etc/profile.d/conda.sh
fi

# Check if environment exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Error: Conda environment '${CONDA_ENV}' does not exist"
    echo "Please run: conda create -n ${CONDA_ENV} python=3.12"
    echo "Then install dependencies: pip install -r requirements.txt"
    exit 1
fi

conda activate ${CONDA_ENV}

# Check if running app or test
if [ "$1" = "test" ]; then
    echo -e "${BLUE}Running test: ${2}${NC}"
    shift  # Remove 'test' argument
    python "$@"
else
    # Start Streamlit app
    echo -e "${GREEN}Starting Streamlit app...${NC}"
    streamlit run app.py
fi

echo "App stopped."
