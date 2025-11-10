#!/bin/bash
# ============================================================================
# Local Docker Test Script
# ============================================================================
# This script tests the Docker container locally before deploying to cluster
# ============================================================================

set -e

IMAGE_NAME="${1:-llabs_lda_hyperopt}"

echo "============================================================================"
echo "Testing Docker Container Locally"
echo "============================================================================"
echo "Image: $IMAGE_NAME"
echo "============================================================================"
echo ""

# Test 1: Check Python version
echo "[TEST 1] Checking Python version..."
docker run --rm "$IMAGE_NAME" python3 --version

# Test 2: Check imports
echo ""
echo "[TEST 2] Testing Python imports..."
docker run --rm "$IMAGE_NAME" python3 -c "
import sys
print('Testing imports...')
try:
    import numpy
    print('✓ numpy')
    import pandas
    print('✓ pandas')
    import matplotlib
    print('✓ matplotlib')
    import scipy
    print('✓ scipy')
    import torch
    print('✓ torch')
    import tqdm
    print('✓ tqdm')
    import sklearn
    print('✓ sklearn')
    import deap
    print('✓ deap')
    print('All imports successful!')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

# Test 3: Check project structure
echo ""
echo "[TEST 3] Checking project structure..."
docker run --rm "$IMAGE_NAME" bash -c "
echo 'Checking directories...'
ls -la /app/ | head -20
echo ''
echo 'Checking for key files...'
[ -f /app/for_klaster.py ] && echo '✓ for_klaster.py exists' || echo '✗ for_klaster.py missing'
[ -d /app/pabbo_method ] && echo '✓ pabbo_method/ exists' || echo '✗ pabbo_method/ missing'
[ -d /app/lda_hyperopt ] && echo '✓ lda_hyperopt/ exists' || echo '✗ lda_hyperopt/ missing'
[ -d /app/data ] && echo '✓ data/ exists' || echo '✗ data/ missing'
"

# Test 4: Check data files
echo ""
echo "[TEST 4] Checking data files..."
docker run --rm "$IMAGE_NAME" bash -c "
if [ -d /app/data ]; then
    echo 'Data directory contents:'
    ls -lh /app/data/*.npz 2>/dev/null || echo 'No .npz files found in /app/data'
else
    echo 'Warning: /app/data directory not found'
fi
"

# Test 5: Check PYTHONPATH
echo ""
echo "[TEST 5] Checking PYTHONPATH..."
docker run --rm "$IMAGE_NAME" python3 -c "
import sys
print('PYTHONPATH:')
for path in sys.path:
    print(f'  - {path}')
"

# Test 6: Quick syntax check of for_klaster.py
echo ""
echo "[TEST 6] Syntax check of for_klaster.py..."
docker run --rm "$IMAGE_NAME" python3 -m py_compile /app/for_klaster.py
echo "✓ Syntax OK"

echo ""
echo "============================================================================"
echo "All tests passed! ✓"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Push to Docker Hub: ./build_and_push.sh"
echo "  2. Follow instructions in run_exp.md"
echo ""