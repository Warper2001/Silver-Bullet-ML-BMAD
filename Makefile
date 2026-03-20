.PHONY: install dev test format lint clean terminal-ui help

# Default target
help:
	@echo "Silver-Bullet-ML-BMAD Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies with Poetry"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run tests (implemented in later stories)"
	@echo "  format      - Format code with black"
	@echo "  lint        - Run flake8 and mypy"
	@echo "  terminal-ui - Launch Rich terminal UI"
	@echo "  clean       - Clean up generated files"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	@echo "Installing dependencies..."
	poetry install

# Install development dependencies
dev: install
	@echo "Installing development dependencies..."
	poetry run pre-commit install

# Run tests (implemented in later stories)
test:
	@echo "Tests will be implemented in later stories"
	@echo "No tests to run yet"

# Format code with black
format:
	@echo "Formatting code with black..."
	poetry run black src/ tests/

# Run linting
lint:
	@echo "Running flake8..."
	poetry run flake8 src/ tests/
	@echo "Running mypy..."
	poetry run mypy src/

# Clean up generated files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/

# Launch terminal UI
terminal-ui:
	@echo "Launching Silver Bullet Terminal UI..."
	@echo "Press Ctrl+C to exit"
	venv/bin/python -m src.monitoring.terminal_ui
	rm -rf build/
