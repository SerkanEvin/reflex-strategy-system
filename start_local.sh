#!/bin/bash
# Start the Local PC (Spinal Cord) for Reflex-Strategy System

# Set environment variables
export VPS_URL="ws://167.86.105.39:8001/ws"  # Updated to port 8001

# Feature flags
export ENABLE_CAPTURE=true
export ENABLE_VISION=true
export ENABLE_INSTINCT=true
export ENABLE_ACTUATOR=false  # Set to false for testing
export ENABLE_VPS_CONNECTION=true

# Debug settings
export DEBUG_MODE=true
export LOG_DETECTIONS=true
export VISUALIZE=false

# Vision
export VISION_MODEL_PATH=
export VISION_CONFIDENCE=0.5
export VISION_DEVICE=cpu

# Instinct
export INSTINCT_CRITICAL_HP=30.0
export INSTINCT_LOW_HP=50.0
export INSTINCT_COMBAT_DISTANCE=150.0

echo "Starting Local PC (Spinal Cord)..."
echo "VPS URL: $VPS_URL"

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

# Start the coordinator
python -m local.coordinator
