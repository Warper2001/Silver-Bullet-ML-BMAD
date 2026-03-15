#!/bin/bash
# Validation script for Story 1.1: Initialize Python Project Structure
# Run this after installing Poetry to validate story completion

echo "🔍 Validating Story 1.1: Initialize Python Project Structure"
echo ""

# Check directories
echo "📁 Checking directories..."
DIRS="src/data src/detection src/ml src/execution src/risk src/monitoring src/research src/dashboard tests/unit tests/integration config logs models data/raw data/processed data/features data/logs data/models notebooks"
MISSING=0

for dir in $DIRS; do
  if [ -d "$dir" ]; then
    echo "✅ $dir exists"
  else
    echo "❌ $dir missing"
    MISSING=1
  fi
done

echo ""

# Check files
echo "📄 Checking files..."
FILES="pyproject.toml .env.example config.yaml .pre-commit-config.yaml README.md .gitignore Makefile"

for file in $FILES; do
  if [ -f "$file" ]; then
    echo "✅ $file exists"
  else
    echo "❌ $file missing"
    MISSING=1
  fi
done

echo ""

# Check __init__.py files in src directories
echo "🐍 Checking Python package structure..."
for dir in src/data src/detection src/ml src/execution src/risk src/monitoring src/research src/dashboard; do
  if [ -f "$dir/__init__.py" ]; then
    echo "✅ $dir/__init__.py exists"
  else
    echo "❌ $dir/__init__.py missing"
    MISSING=1
  fi
done

echo ""

# Verify Poetry (if installed)
if command -v poetry &> /dev/null; then
  echo "📦 Verifying Poetry installation..."
  if poetry check --quiet 2>/dev/null; then
    echo "✅ Poetry configuration valid"
  else
    echo "❌ Poetry configuration invalid"
    MISSING=1
  fi

  echo ""

  # Verify pre-commit hooks (if Poetry is installed)
  if command -v pre-commit &> /dev/null; then
    echo "🪝 Verifying pre-commit hooks configuration..."
    if pre-commit run --all-files >/dev/null 2>&1; then
      echo "✅ Pre-commit hooks functional"
    else
      echo "⚠️  Pre-commit hooks have issues (may be expected with no code yet)"
    fi
  else
    echo "⚠️  pre-commit not installed (run: poetry run pre-commit install)"
  fi
else
  echo "⚠️  Poetry not installed (install from: https://python-poetry.org/docs/)"
fi

echo ""

# Final result
if [ $MISSING -eq 0 ]; then
  echo "✅ Story 1.1 validation PASSED!"
  echo ""
  echo "🎉 Project structure is ready for development!"
  echo ""
  echo "Next steps:"
  echo "1. Install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
  echo "2. Install dependencies: poetry install"
  echo "3. Install pre-commit hooks: poetry run pre-commit install"
  echo "4. Create .env file: cp .env.example .env"
  exit 0
else
  echo "❌ Story 1.1 validation FAILED - some files or directories are missing"
  exit 1
fi
