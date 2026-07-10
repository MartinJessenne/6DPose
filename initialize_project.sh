#!/bin/bash
set -e

echo "===================================================="
echo "6D Pose Project Initialization & Asset Downloader"
echo "===================================================="

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# 1. Hugging Face Login
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face using HF_TOKEN..."
    hf auth login --token "$HF_TOKEN"
else
    echo "HF_TOKEN not detected in environment or .env file."
    # Prompt the user to paste their token directly in the terminal
    read -sp "Please paste your Hugging Face Access Token: " HF_TOKEN_INPUT
    echo "" # Print newline after hidden input
    if [ -n "$HF_TOKEN_INPUT" ]; then
        hf auth login --token "$HF_TOKEN_INPUT"
    else
        echo "No token provided. Proceeding with public download access..."
    fi
fi

# 2. Download YOLO Model Weights
if [ -f best.pt ]; then
    echo -e "\n[YOLO Model] Found locally at 'best.pt'. Skipping download."
else
    echo -e "\n[YOLO Model] Downloading best.pt from Hugging Face..."
    # Download file to a temp folder, copy it to the root path, and clean up
    hf download UItraviolet/yolo_multicart runs/segment/train-2/weights/best.pt --local-dir temp_weights
    mv temp_weights/runs/segment/train-2/weights/best.pt best.pt
    rm -rf temp_weights
    echo "[YOLO Model] Successfully saved to 'best.pt'"
fi

# 3. Download Parquet Dataset
local_dataset_dir="dataset/data"
if [ -d "$local_dataset_dir" ] && ls "$local_dataset_dir"/*.parquet &>/dev/null; then
    echo -e "\n[Dataset] Parquet files already exist in '$local_dataset_dir'. Skipping download."
else
    echo -e "\n[Dataset] Downloading dataset parquet files..."
    mkdir -p "$local_dataset_dir"
    hf download UItraviolet/industrial_cart --repo-type dataset --include "*.parquet" --local-dir "$local_dataset_dir"
    echo "[Dataset] Successfully downloaded dataset parquet files to '$local_dataset_dir'"
fi

echo -e "\nInitialization finished. You can now run the project using:"
echo "  uv run --python 3.12 inspect_pose.py --random 5 --method ransac"
