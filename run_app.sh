#!/bin/bash

# Safe Streamlit launcher for PubMed Research Agent
# Handles potential segfaults and provides better error reporting

echo "=========================================="
echo "PubMed Research Agent - Streamlit Launcher"
echo "=========================================="

# Set environment variables to reduce memory usage and prevent crashes
export GGML_METAL_LOG_LEVEL=0
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=50

# Use the correct Python environment
PYTHON_BIN="/opt/anaconda3/envs/pubmed_py312/bin/python"
STREAMLIT_BIN="/opt/anaconda3/envs/pubmed_py312/bin/streamlit"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found in current directory"
    echo "Please run this script from the project root"
    exit 1
fi

# Check if the environment exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python environment not found at $PYTHON_BIN"
    echo "Please ensure the pubmed_py312 conda environment is installed"
    exit 1
fi

echo ""
echo "Environment: pubmed_py312"
echo "Python: $($PYTHON_BIN --version)"
echo ""
echo "Starting Streamlit app..."
echo "If the app crashes with a segmentation fault, try:"
echo "  1. Selecting 'Rule-based (Fast)' model in the sidebar"
echo "  2. Using a smaller GGUF model"
echo "  3. Checking the error logs above"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Run streamlit with error handling
$STREAMLIT_BIN run app.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 139 ]; then
    echo "Segmentation fault detected (exit code 139)"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. The GGUF model may be too large for available memory"
    echo "  2. Try using 'Rule-based (Fast)' mode instead"
    echo "  3. Reduce n_ctx in agent_gguf.py (currently set to 2048)"
    echo "  4. Set n_gpu_layers=0 in app.py to use CPU only"
    echo ""
elif [ $EXIT_CODE -ne 0 ]; then
    echo "App exited with code: $EXIT_CODE"
else
    echo "App stopped gracefully"
fi
echo "=========================================="

exit $EXIT_CODE
