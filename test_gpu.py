#!/usr/bin/env python3
"""
Quick GPU Test Script
=====================

Быстрая проверка GPU доступности и производительности перед запуском полного pipeline.

Usage:
    python test_gpu.py
"""

import sys
import time
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_gpu_availability():
    """Test 1: Check if GPU is available."""
    print("\n" + "="*80)
    print("TEST 1: GPU Availability")
    print("="*80)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA is NOT available")
            print("\nPossible reasons:")
            print("  1. PyTorch is not installed with CUDA support")
            print("  2. NVIDIA drivers are not installed")
            print("  3. CUDA toolkit is not installed")
            print("\nRecommendation: Use CPU version (for_klaster.py)")
            return False

        print("✓ CUDA is available")
        print(f"✓ Device count: {torch.cuda.device_count()}")
        print(f"✓ Current device: {torch.cuda.current_device()}")
        print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ PyTorch version: {torch.__version__}")

        props = torch.cuda.get_device_properties(0)
        print(f"✓ Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"✓ Compute capability: {props.major}.{props.minor}")

        return True

    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_gpu_memory():
    """Test 2: Check GPU memory."""
    print("\n" + "="*80)
    print("TEST 2: GPU Memory")
    print("="*80)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⊘ Skipped (CUDA not available)")
            return False

        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        free = total_memory - allocated

        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Free: {free:.2f} GB")

        # Check if enough memory
        if total_memory < 8:
            print("\n⚠️  WARNING: Less than 8GB GPU memory detected")
            print("   Recommended: 8GB+ for small model, 16GB+ for large model")
            print("   You may need to reduce batch_size in for_klaster_gpu.py")
            return False
        elif total_memory < 16:
            print("\n✓ Sufficient memory for small model")
            print("⚠️  May be tight for large model with default batch_size")
            return True
        else:
            print("\n✓ Excellent! Sufficient memory for both models")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_gpu_computation():
    """Test 3: Perform simple GPU computation."""
    print("\n" + "="*80)
    print("TEST 3: GPU Computation")
    print("="*80)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⊘ Skipped (CUDA not available)")
            return False

        print("Running matrix multiplication test...")

        # Create random tensors
        size = 5000
        device = torch.device("cuda:0")

        # CPU test
        print(f"  CPU: {size}x{size} matrix multiplication...", end=" ")
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"{cpu_time:.3f}s")

        # GPU test
        print(f"  GPU: {size}x{size} matrix multiplication...", end=" ")
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        torch.cuda.synchronize()
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"{gpu_time:.3f}s")

        speedup = cpu_time / gpu_time
        print(f"\n✓ GPU speedup: {speedup:.1f}x")

        if speedup < 5:
            print("⚠️  WARNING: Low GPU speedup detected")
            print("   This might indicate:")
            print("   - Old GPU")
            print("   - CPU bottleneck")
            print("   - Driver issues")
        else:
            print("✓ Excellent GPU performance!")

        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"\nGPU memory used by test: {allocated:.2f} GB")

        # Clear cache
        del a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()

        return speedup >= 5

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_pabbo_imports():
    """Test 4: Check if PABBO dependencies are available."""
    print("\n" + "="*80)
    print("TEST 4: PABBO Dependencies")
    print("="*80)

    all_ok = True

    # Test imports
    imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
        ("botorch", "BoTorch"),
        ("gpytorch", "GPyTorch"),
    ]

    for module_name, display_name in imports:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {display_name:15s} {version}")
        except ImportError:
            print(f"❌ {display_name:15s} NOT INSTALLED")
            all_ok = False

    if all_ok:
        print("\n✓ All dependencies installed")
    else:
        print("\n❌ Some dependencies missing")
        print("Run: pip install -r requirements.txt")

    return all_ok


def test_pabbo_files():
    """Test 5: Check if PABBO files exist."""
    print("\n" + "="*80)
    print("TEST 5: PABBO Files")
    print("="*80)

    all_ok = True

    files = [
        "pabbo_method/train.py",
        "pabbo_method/evaluate_continuous.py",
        "pabbo_method/policies/transformer.py",
        "pabbo_method/configs/train.yaml",
        "lda_hyperopt/run_no_early_stop.py",
        "lda_hyperopt/optimizers/pabbo_full.py",
        "for_klaster_gpu.py",
    ]

    for file_path in files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_ok = False

    if all_ok:
        print("\n✓ All required files present")
    else:
        print("\n❌ Some files missing")

    return all_ok


def test_data_files():
    """Test 6: Check if data files exist."""
    print("\n" + "="*80)
    print("TEST 6: Data Files")
    print("="*80)

    data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False

    # Look for .npz files
    npz_files = list(data_dir.glob("X_*_val_bow.npz"))

    if not npz_files:
        print("❌ No validation data files found")
        print(f"   Expected: data/X_*_val_bow.npz")
        return False

    print(f"✓ Found {len(npz_files)} dataset(s):")
    for file in npz_files:
        dataset_name = file.stem.replace("X_", "").replace("_val_bow", "")
        file_size = file.stat().st_size / 1e6  # MB
        print(f"  - {dataset_name:20s} ({file_size:.1f} MB)")

    return True


def main():
    """Run all tests."""
    print("="*80)
    print("GPU Test Suite for PABBO Pipeline")
    print("="*80)
    print("This script will check if your system is ready for GPU training.")
    print()

    results = {}

    # Run tests
    results["GPU Available"] = test_gpu_availability()
    results["GPU Memory"] = test_gpu_memory()
    results["GPU Computation"] = test_gpu_computation()
    results["Dependencies"] = test_pabbo_imports()
    results["PABBO Files"] = test_pabbo_files()
    results["Data Files"] = test_data_files()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")

    all_passed = all(results.values())

    print("="*80)

    if all_passed:
        print("\n🎉 SUCCESS! Your system is ready for GPU training!")
        print("\nNext steps:")
        print("  1. Run locally: python for_klaster_gpu.py")
        print("  2. Or submit to cluster: sbatch run_gpu.sh")
        print("\nFor more info, see GPU_SETUP.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")

        if not results["GPU Available"]:
            print("  - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            print("  - Check NVIDIA drivers: nvidia-smi")

        if not results["Dependencies"]:
            print("  - Install requirements: pip install -r requirements.txt")

        if not results["PABBO Files"]:
            print("  - Make sure you're in the project root directory")

        if not results["Data Files"]:
            print("  - Download/place data files in data/ directory")

        print("\nIf GPU is not available, use CPU version: python for_klaster.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())