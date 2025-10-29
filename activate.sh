#!/bin/bash
# Activation script for Prompt Optimizer development environment

echo "🚀 Activating Prompt Optimizer development environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python3 -m venv venv' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Verify Python version
echo "✅ Python version: $(python --version)"

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Core dependencies not installed. Run 'pip install -r requirements.txt'"
fi

if ! python -c "import pytest" 2>/dev/null; then
    echo "⚠️  Development dependencies not installed. Run 'pip install -r requirements-dev.txt'"
fi

echo "🎯 Ready for development! Virtual environment activated."
echo "📁 Project structure:"
echo "   backend/     - Python FastAPI backend"
echo "   frontend/    - Chrome extension files"
echo "   cpp/         - C++ performance modules"
echo "   tests/       - Test suites"
echo "   docs/        - Documentation"
echo "   config/      - Configuration files"
