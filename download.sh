#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the target directory based on the README format
TARGET_DIR="./data/GSCollision"

echo "Creating target directory at ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"

echo "Downloading GSCollision dataset from Hugging Face..."
# Download the dataset directly into the target directory, avoiding symlinks
# so the files are placed physically just as the README describes.
hf download lishiqianhugh/GSCollision \
  --repo-type dataset \
  --local-dir "${TARGET_DIR}" \
  --max-workers 4

echo "=========================================================="
echo "Download complete!"
echo "The dataset has been successfully saved to ${TARGET_DIR}."
echo "It should now contain the expected folders like 'objects', 'backgrounds', and 'scene_configs'."
echo "=========================================================="