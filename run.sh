#!/bin/bash

echo "========================================="
echo "Running LDA Hyperparameter Optimization"
echo "========================================="

mkdir -p logs

echo ""
echo "Results will be saved to: $(pwd)/logs"
echo ""

docker run -v $(pwd)/logs:/app/logs lda-optimizer

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Optimization completed successfully!"
    echo "========================================="
    echo ""
    echo "Results: $(pwd)/logs"
    echo ""
    echo "View TensorBoard:"
    echo "  tensorboard --logdir=logs/"
else
    echo ""
    echo "✗ Optimization failed!"
    exit 1
fi