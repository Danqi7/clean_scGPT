#!/bin/bash

# Clone or update the nano-scGPT repository
REPO_DIR="nano-scGPT"
REPO_URL="https://github.com/Danqi7/nano-scGPT.git"
BRANCH="finetune_prp"

if [ -d "$REPO_DIR" ]; then
    echo "Repository exists. Updating..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    cd ..
else
    echo "Cloning repository..."
    git clone -b "$BRANCH" "$REPO_URL"
fi

# Install the package in development mode
uv pip install -e ./nano-scGPT

echo "nano-scGPT installed successfully!"
echo "You can now use: import nano_scgpt.{}"
