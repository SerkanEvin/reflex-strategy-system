#!/bin/bash
# Start the VPS Brain server for Reflex-Strategy System

# Set environment variables
export VPS_HOST="0.0.0.0"
export VPS_PORT="8001"  # Use 8001 since 8000 is used by Supabase
export DB_HOST="localhost"
export DB_PORT="5432"  # Supabase PostgreSQL internal port
export DB_NAME="reflex_strategy"
export DB_USER="postgres"
export DB_PASSWORD="postgres"
export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_MODEL="gpt-4o-mini"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the server
cd /root/reflex-strategy-system
echo "Starting VPS Brain on port 8001..."
python -m vps.server
