#!/bin/bash
# Database initialization script

echo "🗄️  Initializing Prompt Optimizer Database..."

# Activate virtual environment
source venv/bin/activate

# Create database tables
echo "📊 Creating database tables..."
python -c "
from backend.app.core.database import create_tables
create_tables()
print('✅ Database tables created successfully')
"

echo "🎯 Database initialization complete!"
