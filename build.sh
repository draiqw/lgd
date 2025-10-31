#!/bin/bash

echo "========================================="
echo "Building Docker image: lda-optimizer"
echo "========================================="

docker build -t lda-optimizer .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo "Image 'lda-optimizer' is ready"
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi