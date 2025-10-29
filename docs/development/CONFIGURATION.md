# Configuration Guide

This guide explains how to configure the Prompt Optimizer for different environments.

## Quick Start

1. **Copy the template**: `cp config/env.example .env`
2. **Add your API keys**: Edit `.env` and add your keys
3. **Validate**: Run `./scripts/validate-config.py`

## Environment Variables

### Required Variables

#### API Keys
- `OPENAI_API_KEY`: OpenAI API key for embeddings (get from [OpenAI](https://platform.openai.com/api-keys))
- `GROQ_API_KEY`: Groq API key for optimization (get from [Groq](https://console.groq.com))

### Optional Variables

#### Application Settings
- `ENVIRONMENT`: Current environment (`development`, `production`, `test`)
- `DEBUG`: Enable debug mode (`true`/`false`)
- `SECRET_KEY`: Secret key for session management (change in production!)

#### Server Settings
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts

#### Database
- `DATABASE_URL`: Database connection string
  - SQLite (default): `sqlite:///./prompt_optimizer.db`
  - PostgreSQL: `postgresql://user:password@host:port/dbname`

#### Redis (Optional)
- `REDIS_URL`: Redis connection string (default: `redis://localhost:6379`)
- `REDIS_ENABLED`: Enable Redis caching (`true`/`false`)

#### Embedding Settings
- `EMBEDDING_MODEL`: Model name (default: `text-embedding-3-small`)
- `EMBEDDING_DIMENSION`: Dimension size (default: `1536`)
- `MAX_SIMILAR_PROMPTS`: Max similar prompts to retrieve (default: `5`)

#### C++ Module (Optional)
- `CPP_MODULE_ENABLED`: Enable C++ performance module (`true`/`false`)
- `CPP_MODULE_PATH`: Path to compiled module (default: `./cpp/build/similarity.so`)

#### Caching
- `CACHE_TTL_SECONDS`: Cache time-to-live (default: `3600`)
- `CACHE_ENABLED`: Enable caching (`true`/`false`)

#### Rate Limiting
- `RATE_LIMIT_PER_MINUTE`: Max requests per minute (default: `60`)

## Environment-Specific Configuration

### Development
```bash
ENVIRONMENT=development
DEBUG=true
DATABASE_ECHO=true
ALLOWED_HOSTS=*
REDIS_ENABLED=false
```

### Production
```bash
ENVIRONMENT=production
DEBUG=false
DATABASE_ECHO=false
ALLOWED_HOSTS=your-domain.com
REDIS_ENABLED=true
SECRET_KEY=<generate-strong-secret-key>
```

### Test
```bash
ENVIRONMENT=test
DEBUG=true
DATABASE_URL=sqlite:///./test_prompt_optimizer.db
REDIS_ENABLED=false
CACHE_ENABLED=false
CPP_MODULE_ENABLED=false
```

## Configuration Scripts

### Setup Configuration
Interactive setup from template:
```bash
python scripts/setup-config.py
```

### Validate Configuration
Check your configuration:
```bash
python scripts/validate-config.py
```

### Initialize Database
Set up database tables:
```bash
./scripts/init-db.sh
```

## Validation

The configuration system validates:
- ✅ Required API keys are set
- ✅ File paths exist
- ✅ Database is accessible
- ✅ Redis is accessible (if enabled)
- ✅ C++ module exists (if enabled)
- ✅ Production security settings

## Security Best Practices

### API Keys
1. **Never commit `.env` to version control**
2. Use environment-specific keys when possible
3. Rotate keys periodically
4. Use least-privilege API keys

### Production Settings
- Set `DEBUG=false`
- Change `SECRET_KEY` to a strong random value
- Use production database
- Enable Redis caching
- Set proper `ALLOWED_HOSTS`

### Secret Key Generation
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Troubleshooting

### Missing API Keys
```bash
# Check which keys are missing
python scripts/validate-config.py

# Add to .env file
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

### Redis Connection Failed
```bash
# Check if Redis is running
redis-cli ping

# Install and start Redis (macOS)
brew install redis
brew services start redis
```

### C++ Module Not Found
The C++ module is optional and will be built in Phase 3. For now, disable it:
```bash
CPP_MODULE_ENABLED=false
```

## Configuration Files

- `.env` - Your local configuration (not in git)
- `config/env.example` - Template file
- `config/dev/config.env` - Development defaults
- `config/prod/config.env` - Production defaults
- `config/test/config.env` - Test defaults

## Next Steps

After configuration:
1. Validate: `python scripts/validate-config.py`
2. Initialize DB: `./scripts/init-db.sh`
3. Start server: `python main.py`
