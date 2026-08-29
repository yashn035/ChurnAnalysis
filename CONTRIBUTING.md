# Contributing to Customer Churn Analysis & Retention System 🤝

Issues and PRs welcome! Please run `pre-commit run --all-files` before submitting.

---

## 🚀 How to Contribute

We welcome contributions from the community! Follow these steps to set up your development environment and submit pull requests.

### 1. Fork & Clone the Repository

```bash
git clone https://github.com/yashn035/ChurnAnalysis.git
cd ChurnAnalysis
```

### 2. Set Up Virtual Environment & Dependencies

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Install Pre-commit Hooks

Ensure code formatting (`black`), import sorting (`isort`), linting (`flake8`), and file cleanup standards are automatically enforced:

```bash
pre-commit install
```

### 4. Code Standards & Verification

Before submitting a pull request, run the following checks locally:

```bash
# 1. Run unit test suite
make test   # or: pytest tests/

# 2. Run code quality pre-commit checks
pre-commit run --all-files

# 3. Test master execution launcher
python src/launch_all.py
```

### 5. Submit Pull Request

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Commit your changes: `git commit -m "Add descriptive commit message"`
3. Push to your fork: `git push origin feature/your-feature-name`
4. Open a Pull Request on GitHub describing your changes.

Thank you for contributing! ❤️
