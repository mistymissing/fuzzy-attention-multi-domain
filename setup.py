"""
Setup script for UniMatch-Clip package
"""

from setuptools import setup, find_packages

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements file
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="unimatch-clip",
    version="1.0.0",
    author="UniMatch-Clip Contributors",
    author_email="your.email@domain.com",
    description="Fuzzy Attention-Based Multi-Domain Learning with Difficulty-Aware Processing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/UniMatch-Clip",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "unimatch-train=src.train:main",
            "unimatch-api=api_server:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/UniMatch-Clip/issues",
        "Source": "https://github.com/your-username/UniMatch-Clip",
    },
    keywords="multi-domain learning, cross-domain, attention mechanism, deep learning, pytorch",
    include_package_data=True,
    zip_safe=False,
)