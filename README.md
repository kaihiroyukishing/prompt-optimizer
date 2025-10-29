# Prompt Optimizer

A context-aware AI prompt optimization system with a Python FastAPI backend and C++ performance modules. This project transforms simple prompt optimization into an intelligent, context-aware system that learns from past interactions.

## 🎯 Project Overview

The Prompt Optimizer consists of three main components:

1. **Chrome Extension** (Frontend) - User interface for prompt optimization
2. **Python Backend** (FastAPI) - Handles API requests, context storage, and AI integration
3. **C++ Module** (Performance) - High-performance similarity calculations and caching

## 🏗️ Architecture

```
Chrome Extension (JavaScript + UI)
           ↓
Python Backend (FastAPI)
├── Context DB (SQLite)
├── Groq API Integration
├── Embedding Storage (FAISS)
└── Calls to C++ Engine
           ↓
C++ Module (pybind11)
├── Cosine Similarity
├── Binary Caching Layer
└── RAII + STL for Performance
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js (for Chrome extension development)
- C++ compiler (for performance modules)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/prompt-optimizer.git
   cd prompt-optimizer
   ```

2. **Set up the development environment**
   ```bash
   ./activate.sh
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Configure environment variables**
   ```bash
   cp config/env.example .env
   # Edit .env with your API keys
   ```

5. **Initialize the database**
   ```bash
   ./scripts/init-db.sh
   ```

6. **Start the development server**
   ```bash
   ./scripts/dev-server.sh
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 📁 Project Structure

```
promptOptimizer/
├── backend/                 # Python FastAPI backend
│   ├── app/                # Main application code
│   │   ├── core/          # Core functionality (config, database)
│   │   └── routes/        # API route handlers
│   ├── models/            # Database models
│   ├── services/          # Business logic
│   └── utils/             # Helper functions
├── frontend/               # Chrome extension
│   ├── js/                # JavaScript files
│   ├── css/               # Stylesheets
│   └── assets/            # Images, icons, etc.
├── cpp/                   # C++ performance modules
│   ├── src/               # Source files
│   └── include/           # Header files
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   └── integration/        # Integration tests
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # Build/deployment scripts
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml        # Project metadata
└── main.py               # Application entry point
```

## 🔧 Development

### Environment Setup

The project uses a virtual environment for dependency isolation:

```bash
# Activate environment
./activate.sh

# Or manually
source venv/bin/activate
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test types
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black backend/

# Sort imports
isort backend/

# Lint code
flake8 backend/

# Type checking
mypy backend/
```

### Database Operations

```bash
# Initialize database
./scripts/init-db.sh

# Reset database (development)
rm prompt_optimizer.db
./scripts/init-db.sh
```

## 🌐 API Endpoints

### Optimization
- `POST /api/v1/optimize` - Optimize a prompt with context
- `GET /api/v1/prompts/{session_id}` - Get session prompts

### Context Management
- `POST /api/v1/save-context` - Save prompt context
- `GET /api/v1/context/{session_id}` - Get session context
- `DELETE /api/v1/context/{session_id}` - Clear session context

### Health & Info
- `GET /` - API information
- `GET /health` - Health check

## 🔑 Configuration

Environment variables are managed through `.env` files:

- `config/env.example` - Template with all available options
- `config/dev/config.env` - Development settings
- `config/prod/config.env` - Production settings
- `config/test/config.env` - Test settings

Key configuration options:
- `OPENAI_API_KEY` - OpenAI API key for embeddings
- `GROQ_API_KEY` - Groq API key for optimization
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection (optional)
- `CPP_MODULE_ENABLED` - Enable C++ performance modules

## 🧪 Testing

The project includes comprehensive testing:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test API endpoints and database interactions
- **Performance Tests**: Benchmark C++ vs Python implementations

Run tests with:
```bash
pytest tests/
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running the server
- **Development Guide**: See `docs/development/`
- **User Guide**: See `docs/user/`

## 🚀 Deployment

### Development
```bash
./scripts/dev-server.sh
```

### Production
```bash
# Set environment
export ENVIRONMENT=production

# Start production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Check the `/docs` endpoint when running the server
- **Discussions**: Use GitHub Discussions for questions and ideas

## 🔮 Roadmap

- [ ] Phase 1: Backend Setup (Python + FastAPI)
- [ ] Phase 2: Embedding System
- [ ] Phase 3: C++ Integration for Performance
- [ ] Phase 4: Chrome Extension Integration
- [ ] Phase 5: Testing & Metrics

---

**Built with ❤️ using FastAPI, SQLAlchemy, OpenAI, Groq, FAISS, and C++**
