# 🤝 Contributing to BindingRMSD

We welcome contributions to BindingRMSD! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Code Style](#code-style)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/BindingRMSD.git
   cd BindingRMSD
   ```
3. Set up the development environment (see [Development Setup](#development-setup))
4. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- Conda (recommended for environment management)

### Installation

1. Create a conda environment:
   ```bash
   conda create -n bindingrmsd-dev python=3.11
   conda activate bindingrmsd-dev
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Install development dependencies:
   ```bash
   pip install pytest black flake8 isort mypy
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=bindingrmsd
```

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code with black
black bindingrmsd/

# Sort imports with isort
isort bindingrmsd/

# Check code style with flake8
flake8 bindingrmsd/

# Type checking with mypy
mypy bindingrmsd/
```

## 📝 Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug fixes**: Fix issues and improve stability
- ✨ **New features**: Add new functionality or improve existing features
- 📚 **Documentation**: Improve docs, examples, or tutorials
- 🎨 **Code improvements**: Refactor code for better performance or maintainability
- 🧪 **Tests**: Add or improve test coverage

### What to Contribute

**High Priority:**
- Performance optimizations
- Memory usage improvements
- Support for additional molecular file formats
- Enhanced model architectures
- Better error handling and logging

**Medium Priority:**
- Code refactoring and cleanup
- Additional unit tests
- Documentation improvements
- Example notebooks and tutorials

**Low Priority:**
- Minor feature additions
- Cosmetic changes

## 🔄 Pull Request Process

1. **Before You Start:**
   - Check existing issues and PRs to avoid duplicates
   - Open an issue to discuss major changes
   - Ensure your changes align with project goals

2. **Making Changes:**
   - Create a feature branch from `main`
   - Make your changes in logical, atomic commits
   - Write clear commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Testing:**
   - Run all tests locally
   - Ensure code passes linting checks
   - Test on both CPU and GPU if possible
   - Verify backward compatibility

4. **Submitting:**
   - Push your branch to your fork
   - Create a pull request with a clear description
   - Reference any related issues
   - Be responsive to feedback and review comments

### Commit Message Format

Use clear, descriptive commit messages:

```
Add support for MOL2 file format

- Implement MOL2 parser in data/utils.py
- Add MOL2 format detection
- Update tests and documentation
- Fixes #123
```

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment:** OS, Python version, package versions
- **Steps to reproduce:** Minimal example that reproduces the issue
- **Expected behavior:** What should happen
- **Actual behavior:** What actually happens
- **Error messages:** Full error traceback if applicable
- **Additional context:** Any other relevant information

### Bug Report Template

```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.11.0]
- BindingRMSD: [e.g., 0.1.0]
- PyTorch: [e.g., 2.4.0]
- CUDA: [e.g., 12.1]

**Describe the bug:**
A clear description of what the bug is.

**To Reproduce:**
Steps to reproduce the behavior:
1. Run command '...'
2. Use input file '...'
3. See error

**Expected behavior:**
A clear description of what you expected to happen.

**Error message:**
```
[Paste full error traceback here]
```

**Additional context:**
Add any other context about the problem here.
```

## 💡 Feature Requests

We welcome feature requests! Please provide:

- **Clear description:** What feature would you like to see?
- **Use case:** Why is this feature needed?
- **Implementation ideas:** Any thoughts on how to implement it?
- **Alternatives:** Have you considered any alternatives?

## 🎨 Code Style

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Use meaningful variable and function names
- Keep functions focused and concise

### Documentation Style

- Use clear, concise language
- Include examples where helpful
- Update relevant documentation for changes
- Use proper markdown formatting

### Example Code Style

```python
def predict_rmsd(
    protein_pdb: str,
    ligand_file: str,
    model_path: str,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Predict RMSD for protein-ligand binding poses.
    
    Args:
        protein_pdb: Path to protein PDB file
        ligand_file: Path to ligand file
        model_path: Path to model weights directory
        device: Compute device ('cpu' or 'cuda')
        
    Returns:
        DataFrame containing prediction results
        
    Raises:
        FileNotFoundError: If input files are not found
        ValueError: If invalid parameters are provided
    """
    # Implementation here
    pass
```

## 🏷️ Issue and PR Labels

We use labels to categorize issues and PRs:

- `bug`: Something isn't working
- `feature`: New feature or request
- `documentation`: Improvements to docs
- `performance`: Performance improvements
- `testing`: Related to testing
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## 📞 Getting Help

If you need help with contributing:

- Check existing issues and discussions
- Open an issue with the `question` label
- Reach out to maintainers

## 🙏 Recognition

Contributors will be acknowledged in:

- The project's README
- Release notes for significant contributions
- Special recognition for outstanding contributions

Thank you for contributing to BindingRMSD! 🎉 