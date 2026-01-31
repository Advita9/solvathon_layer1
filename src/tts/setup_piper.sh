#!/bin/bash

# Directory setup
mkdir -p src/tts/piper_bin
mkdir -p src/tts/piper_models

# 1. Pipeline binary is installed via pip (piper-tts)
# We only need to download models now.

echo "✅ Piper is installed via pip."

# 2. Download Voice Models
MODELS_DIR="src/tts/piper_models"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main"

download_model() {
    LANG=$1
    REGION=$2
    NAME=$3
    QUALITY=$4
    
    FILE="${LANG}_${REGION}-${NAME}-${QUALITY}.onnx"
    JSON="${LANG}_${REGION}-${NAME}-${QUALITY}.onnx.json"
    
    # URL Pattern: lang/lang_REGION/name/quality/filename
    # e.g. en/en_US/lessac/medium/en_US-lessac-medium.onnx
    
    echo "⬇️  Downloading Voice: $FILE"
    curl -L -o "$MODELS_DIR/$FILE" "$BASE_URL/$LANG/${LANG}_${REGION}/$NAME/$QUALITY/$FILE"
    curl -L -o "$MODELS_DIR/$JSON" "$BASE_URL/$LANG/${LANG}_${REGION}/$NAME/$QUALITY/$JSON"
}

# English
download_model "en" "US" "lessac" "medium"

# Hindi (Try Dhananjai x_low or Rohan)
echo "⬇️  Attempting Hindi Download (Dhananjai x_low)..."
download_model "hi" "IN" "dhananjai" "x_low"
# If x_low also small (error), you might check manually.

# Telugu (Request: Maya)
download_model "te" "IN" "maya" "medium"

echo "✅ Piper Setup Complete!"
ls -l $MODELS_DIR
