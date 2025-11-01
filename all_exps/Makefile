# ============================================================================
# Makefile for Optimization Algorithms Testing
# ============================================================================
#
# Usage:
#   make test                  - Run all algorithms test (CPU)
#   make test-gpu              - Run all algorithms test (GPU - for PABBO with Transformer)
#   make train-pabbo          - Train PABBO Transformer model
#   make clean                 - Clean test results
#   make install               - Install dependencies
#   make help                  - Show this help message
#
# ============================================================================

# Python interpreter
PYTHON := python3

# Virtual environment
VENV := ../.venv
ACTIVATE := source $(VENV)/bin/activate

# Directories
RESULTS_DIR := test_results
PABBO_DIR := ../pabbo_method
LOGS_DIR := test_results

# Configuration
SEED := 42
ITERATIONS := 50
POP_SIZE := 20

.PHONY: all help install test test-gpu train-pabbo clean visualize check-deps

# Default target
all: help

# Help message
help:
	@echo "============================================================================"
	@echo "Optimization Algorithms Testing - Makefile"
	@echo "============================================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make test           - Run all algorithms test (GA, ES, PABBO)"
	@echo "  make test-gpu       - Run test on GPU (for future PABBO Transformer)"
	@echo "  make train-pabbo    - Train PABBO Transformer model (~10-20 min)"
	@echo "  make clean          - Clean all test results and logs"
	@echo "  make install        - Install all Python dependencies"
	@echo "  make visualize      - Visualize test results"
	@echo "  make check-deps     - Check if all dependencies are installed"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "Environment:"
	@echo "  Python: $(PYTHON)"
	@echo "  Virtual Env: $(VENV)"
	@echo "  Results Dir: $(RESULTS_DIR)"
	@echo ""
	@echo "Configuration:"
	@echo "  Seed: $(SEED)"
	@echo "  Iterations: $(ITERATIONS)"
	@echo "  Population size: $(POP_SIZE)"
	@echo ""
	@echo "============================================================================"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install numpy matplotlib scipy deap
	$(PYTHON) -m pip install torch torchvision botorch gpytorch
	$(PYTHON) -m pip install hydra-core omegaconf wandb
	@echo "✅ Dependencies installed successfully!"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import numpy; print('✓ numpy:', numpy.__version__)" || echo "✗ numpy not found"
	@$(PYTHON) -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)" || echo "✗ matplotlib not found"
	@$(PYTHON) -c "import scipy; print('✓ scipy:', scipy.__version__)" || echo "✗ scipy not found"
	@$(PYTHON) -c "import deap; print('✓ deap:', deap.__version__)" || echo "✗ deap not found"
	@$(PYTHON) -c "import torch; print('✓ torch:', torch.__version__)" || echo "✗ torch not found"
	@echo "✅ Dependency check complete!"

# Run test on CPU
test: check-deps
	@echo "============================================================================"
	@echo "Running Optimization Algorithms Test (CPU)"
	@echo "============================================================================"
	@echo "Configuration:"
	@echo "  - Test function: Non-differentiable with discontinuities"
	@echo "  - Algorithms: GA, ES, PABBO (simplified)"
	@echo "  - Iterations: $(ITERATIONS)"
	@echo "  - Population size: $(POP_SIZE)"
	@echo "  - Seed: $(SEED)"
	@echo ""
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) test.py 2>&1 | tee $(LOGS_DIR)/test_run.log
	@echo ""
	@echo "============================================================================"
	@echo "✅ Test completed! Results saved to:"
	@echo "   - Function plot: $(RESULTS_DIR)/test_function.png"
	@echo "   - Convergence plots: $(RESULTS_DIR)/convergence.png"
	@echo "   - Logs:"
	@echo "       * $(LOGS_DIR)/GA_test.log"
	@echo "       * $(LOGS_DIR)/ES_test.log"
	@echo "       * $(LOGS_DIR)/PABBO_test.log"
	@echo "       * $(LOGS_DIR)/test_run.log"
	@echo "============================================================================"

# Run test on GPU (for future PABBO Transformer)
test-gpu: check-deps
	@echo "============================================================================"
	@echo "Running Optimization Algorithms Test (GPU)"
	@echo "============================================================================"
	@echo "⚠️  Note: GPU support is for PABBO Transformer only (not yet implemented)"
	@echo "   Current test will run on CPU"
	@echo ""
	@mkdir -p $(RESULTS_DIR)
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) test.py 2>&1 | tee $(LOGS_DIR)/test_run_gpu.log
	@echo ""
	@echo "✅ GPU test completed!"

# Train PABBO Transformer model
train-pabbo: check-deps
	@echo "============================================================================"
	@echo "Training PABBO Transformer Model"
	@echo "============================================================================"
	@echo "Configuration:"
	@echo "  - Function: rastrigin1d_test"
	@echo "  - Steps: 2000 (reduced for quick training)"
	@echo "  - Device: CPU (change to cuda in config if GPU available)"
	@echo "  - Estimated time: ~10-20 minutes on CPU"
	@echo ""
	@echo "Starting training..."
	cd $(PABBO_DIR) && $(PYTHON) train.py --config-name train_rastrigin1d_test
	@echo ""
	@echo "============================================================================"
	@echo "✅ PABBO model training completed!"
	@echo "   Model saved to: $(PABBO_DIR)/results/PABBO/..."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Check $(PABBO_DIR)/results/ for the trained model"
	@echo "  2. Update test.py to use the trained model (see PABBO_PRETRAIN_GUIDE.md)"
	@echo "============================================================================"

# Visualize results
visualize:
	@echo "Opening visualization..."
	@if [ -f "$(RESULTS_DIR)/convergence.png" ]; then \
		open $(RESULTS_DIR)/convergence.png || xdg-open $(RESULTS_DIR)/convergence.png; \
	else \
		echo "❌ No convergence plot found. Run 'make test' first."; \
	fi
	@if [ -f "$(RESULTS_DIR)/test_function.png" ]; then \
		open $(RESULTS_DIR)/test_function.png || xdg-open $(RESULTS_DIR)/test_function.png; \
	else \
		echo "❌ No function plot found. Run 'make test' first."; \
	fi

# Clean results
clean:
	@echo "Cleaning test results..."
	rm -rf $(RESULTS_DIR)
	rm -f *.pyc
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "✅ Clean complete!"

# Quick test (fewer iterations)
quick-test:
	@echo "Running quick test (10 iterations)..."
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) -c "\
import sys; \
from test import main; \
main()"
	@echo "✅ Quick test complete!"

# Show logs
logs:
	@echo "============================================================================"
	@echo "Recent Test Logs"
	@echo "============================================================================"
	@if [ -f "$(LOGS_DIR)/test_run.log" ]; then \
		tail -n 50 $(LOGS_DIR)/test_run.log; \
	else \
		echo "No logs found. Run 'make test' first."; \
	fi

# Show results summary
summary:
	@echo "============================================================================"
	@echo "Test Results Summary"
	@echo "============================================================================"
	@if [ -f "$(LOGS_DIR)/test_run.log" ]; then \
		grep -A 10 "SUMMARY" $(LOGS_DIR)/test_run.log || echo "No summary found"; \
	else \
		echo "No results found. Run 'make test' first."; \
	fi

# Development: run with python debugger
debug:
	@echo "Running test with debugger..."
	$(PYTHON) -m pdb test.py