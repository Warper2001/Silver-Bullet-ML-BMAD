#!/bin/bash
# Validation script for Story 1.2: Implement TradeStation API Authentication
# Run this after Poetry installation completes

echo "🔍 Validating Story 1.2: Implement TradeStation API Authentication"
echo ""

# Check if Poetry installation is complete
if [ ! -f "poetry.lock" ]; then
    echo "❌ Poetry installation not complete yet"
    echo "Please wait for 'poetry install' to finish, then run this script again"
    exit 1
fi

echo "✅ Poetry installation complete"
echo ""

# Check files created
echo "📄 Checking implementation files..."
FILES=(
    "src/data/exceptions.py"
    "src/data/config.py"
    "src/data/auth.py"
    "tests/unit/test_auth.py"
    "tests/integration/test_auth_integration.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo ""

# Run tests
echo "🧪 Running tests..."
poetry run pytest tests/unit/test_auth.py -v
if [ $? -eq 0 ]; then
    echo "✅ Unit tests passed"
else
    echo "❌ Unit tests failed"
    exit 1
fi

echo ""

poetry run pytest tests/integration/test_auth_integration.py -v
if [ $? -eq 0 ]; then
    echo "✅ Integration tests passed"
else
    echo "❌ Integration tests failed"
    exit 1
fi

echo ""

# Type checking
echo "🔍 Running type checking..."
poetry run mypy src/data/
if [ $? -eq 0 ]; then
    echo "✅ Type checking passed"
else
    echo "❌ Type checking failed"
    exit 1
fi

echo ""

# Code formatting
echo "🎨 Checking code formatting..."
poetry run black --check src/data tests/
if [ $? -eq 0 ]; then
    echo "✅ Code formatting correct"
else
    echo "⚠️  Code formatting issues found, running black..."
    poetry run black src/data tests/
fi

echo ""

# Linting
echo "🔍 Running linting..."
poetry run flake8 src/data tests/ --ignore=E501,W503
if [ $? -eq 0 ]; then
    echo "✅ Linting passed"
else
    echo "❌ Linting failed"
    exit 1
fi

echo ""
echo "✅ Story 1.2 validation PASSED!"
echo ""
echo "🎉 Implementation complete!"
echo ""
echo "Next steps:"
echo "1. Review the implementation"
echo "2. Run code-review: code-review 1-2-implement-tradestation-api-authentication"
echo "3. Proceed to Story 1.3: Establish WebSocket Connection"
