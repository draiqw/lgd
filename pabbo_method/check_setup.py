#!/usr/bin/env python
"""
Check PABBO installation and setup.

This script verifies that:
1. All required dependencies are installed
2. All modules can be imported correctly
3. Configuration files are present
4. Directory structure is correct

Run this before training to catch any setup issues.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.8"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python {version.major}.{version.minor} (need >= 3.8)")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required packages"""
    print("\nChecking dependencies...")

    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'botorch': 'BoTorch',
        'gpytorch': 'GPyTorch',
        'hydra': 'Hydra',
        'omegaconf': 'OmegaConf',
        'matplotlib': 'Matplotlib',
    }

    optional_packages = {
        'wandb': 'Weights & Biases (optional, for logging)',
        'tensorboardX': 'TensorBoardX (optional)',
    }

    all_ok = True

    # Check required
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False

    # Check optional
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - not installed (optional)")

    return all_ok


def check_modules():
    """Check PABBO modules can be imported"""
    print("\nChecking PABBO modules...")

    modules = [
        ('wandb_wrapper', 'wandb_wrapper'),
        ('utils.log', 'Utils: logging'),
        ('utils.losses', 'Utils: losses'),
        ('utils.paths', 'Utils: paths'),
        ('data.sampler', 'Data: sampler'),
        ('data.function', 'Data: functions'),
        ('data.environment', 'Data: environment'),
        ('policies.transformer', 'Policies: Transformer'),
        ('policy_learning', 'Policy learning'),
    ]

    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} - {e}")
            all_ok = False

    return all_ok


def check_configs():
    """Check configuration files"""
    print("\nChecking configuration files...")

    config_dir = Path('configs')
    if not config_dir.exists():
        print(f"  ✗ configs/ directory missing")
        return False

    required_configs = [
        'train.yaml',
        'evaluate.yaml',
        'train_rastrigin1d_test.yaml',
    ]

    all_ok = True
    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  ✓ {config_file}")
        else:
            print(f"  ✗ {config_file} - missing")
            all_ok = False

    return all_ok


def check_structure():
    """Check directory structure"""
    print("\nChecking directory structure...")

    required_dirs = [
        'configs',
        'data',
        'policies',
        'utils',
    ]

    required_files = [
        'train.py',
        'evaluate_continuous.py',
        'policy_learning.py',
        'wandb_wrapper.py',
        'requirements.txt',
    ]

    all_ok = True

    for dirname in required_dirs:
        dirpath = Path(dirname)
        if dirpath.exists() and dirpath.is_dir():
            print(f"  ✓ {dirname}/")
        else:
            print(f"  ✗ {dirname}/ - missing")
            all_ok = False

    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - missing")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability (optional)"""
    print("\nChecking CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ⚠ Could not check CUDA: {e}")


def main():
    """Run all checks"""
    print("=" * 70)
    print("PABBO Setup Check")
    print("=" * 70)

    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("PABBO modules", check_modules),
        ("Config files", check_configs),
        ("Directory structure", check_structure),
    ]

    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False

    # Optional checks (don't affect pass/fail)
    check_cuda()

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All checks passed! You're ready to train PABBO.")
        print("\nQuick start:")
        print("  python train.py --config-name=train_rastrigin1d_test")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())