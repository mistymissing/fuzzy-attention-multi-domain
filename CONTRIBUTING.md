# Contributing to UniMatch-Clip

Thank you for your interest in contributing to UniMatch-Clip! We welcome contributions from the community and are pleased to have you join us.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that the bug has already been reported. When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Provide specific examples to demonstrate the steps
- Describe the current behavior and explain which behavior you expected to see instead
- Explain why this enhancement would be useful

### Pull Requests

The process described here has several goals:

- Maintain UniMatch-Clip's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible UniMatch-Clip
- Enable a sustainable system for maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in the template
2. Follow the styleguides
3. After you submit your pull request, verify that all status checks are passing

## Development Environment Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Conda (recommended)

### Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/UniMatch-Clip.git
cd UniMatch-Clip
```

2. Create a conda environment:
```bash
conda create -n unimatch-dev python=3.8
conda activate unimatch-dev
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Install development dependencies:
```bash
pip install pytest black flake8 mypy
```

5. Verify installation:
```bash
python verify_project.py
```

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Python Styleguide

We follow PEP 8 with some modifications:

- Use Black for code formatting
- Line length: 88 characters
- Use type hints where possible
- Document functions and classes with docstrings

Format your code before submitting:
```bash
black src/
flake8 src/
mypy src/
```

### Documentation Styleguide

- Use Markdown for documentation
- Keep lines under 80 characters when possible
- Use clear, concise language
- Include code examples where appropriate

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_adapters.py
```

### Writing Tests

- Write tests for new features and bug fixes
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies

## Project Structure

Familiarize yourself with the project structure:

```
src/
├── models/          # Core model implementations
├── data/           # Data loading and preprocessing
├── utils/          # Utility functions
└── train.py        # Training script

experiments/        # Experimental scripts
configs/           # Configuration files
tests/            # Test files
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is negative.
    """
    pass
```

### Adding New Features

When adding new features:

1. Add appropriate tests
2. Update documentation
3. Update configuration files if needed
4. Add examples to README if applicable

## Getting Help

If you need help, you can:

- Check the documentation
- Look at existing issues
- Create a new issue with the "question" label
- Contact the maintainers

## Recognition

Contributors will be recognized in:

- The README file
- Release notes for significant contributions
- The project's documentation

Thank you for contributing to UniMatch-Clip!