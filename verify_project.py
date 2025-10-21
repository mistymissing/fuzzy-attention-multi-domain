#!/usr/bin/env python3
"""
UniMatch-Clip Project Validation Script
Validates project structure and basic dependencies are correctly installed
"""

import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python version too low, requires Python 3.8+")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_project_structure():
    """Check project structure"""
    required_paths = [
        "src/models/adapters.py",
        "src/models/fuzzy_attention.py",
        "src/models/difficulty_aware.py",
        "src/data/dataloader.py",
        "src/utils/metrics.py",
        "src/train.py",
        "configs/config.yaml",
        "requirements.txt",
        "README.md"
    ]

    missing_files = []
    for path in required_paths:
        if not Path(path).exists():
            missing_files.append(path)

    if missing_files:
        print("❌ Missing the following files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    print("✅ Project structure complete")
    return True

def check_dependencies():
    """Check key dependencies"""
    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("sklearn", "Scikit-learn")
    ]

    missing_deps = []
    for module, name in dependencies:
        try:
            importlib.import_module(module)
            print(f"✅ {name}")
        except ImportError:
            missing_deps.append(name)
            print(f"❌ {name} not installed")

    return len(missing_deps) == 0

def check_conda_env():
    """Check if running in conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ Conda environment: {conda_env}")
        if conda_env == 'er':
            print("✅ Using recommended 'er' environment")
        return True
    else:
        print("⚠️  No conda environment detected, recommend using conda environment 'er'")
        return False

def main():
    """Main validation function"""
    print("🔍 UniMatch-Clip Project Validation")
    print("=" * 50)

    checks = [
        ("Python Version Check", check_python_version),
        ("Project Structure Check", check_project_structure),
        ("Conda Environment Check", check_conda_env),
        ("Dependencies Check", check_dependencies)
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        result = check_func()
        if not result:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Project validation passed! Ready to use UniMatch-Clip")
        print("\n📚 Next steps you can take:")
        print("   1. Run pip install -r requirements.txt to install all dependencies")
        print("   2. Run python src/train.py to start training")
        print("   3. Run python api_server.py to start API service")
        print("   4. Run python interactive_demo.py to view demonstration")
    else:
        print("⚠️  Project validation found issues, please fix according to above prompts")
        print("\n💡 Suggestions:")
        print("   1. Activate conda environment: conda activate er")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Re-run validation: python verify_project.py")

if __name__ == "__main__":
    main()