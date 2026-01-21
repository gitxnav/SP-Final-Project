#!/bin/bash

# Run Streamlit Application for CKD Detection
# Usage: ./run_app.sh

echo "🏥 Starting CKD Detection Dashboard..."
echo "=================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed!"
    echo "Installing streamlit..."
    pip install streamlit==1.29.0
fi

# Navigate to src directory and run app
cd "$(dirname "$0")/"

echo "📂 Current directory: $(pwd)"
echo "🚀 Launching Streamlit application..."
echo ""

streamlit run app.py --server.port 8502

# Alternative: Run with custom port
# streamlit run src/app.py --server.port 8502
