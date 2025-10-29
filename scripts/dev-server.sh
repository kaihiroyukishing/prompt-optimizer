#!/bin/bash
# Development server startup script

echo "🚀 Starting Prompt Optimizer Development Server..."

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from template..."
    cp config/env.example .env
    echo "📝 Please edit .env file with your API keys and configuration"
fi

# Start the development server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API documentation available at http://localhost:8000/docs"
echo "🔄 Auto-reload enabled for development"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
