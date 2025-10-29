#!/usr/bin/env python3
"""
Interactive configuration setup script for Prompt Optimizer

This script helps users set up their environment configuration interactively.
"""

import sys
from pathlib import Path

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from backend.app.core.config import get_settings
from backend.utils.config_helpers import (
    create_env_file_from_template,
    print_configuration_summary,
    validate_environment_setup,
)

# Configure logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger()


def main():
    print("🚀 Prompt Optimizer Configuration Setup")
    print("=" * 60)

    # Check if .env already exists
    env_file = Path(".env")
    if env_file.exists():
        response = input(
            "\n.env file already exists. Do you want to recreate it? (yes/no): "
        )
        if response.lower() not in ["yes", "y"]:
            print("Keeping existing .env file.")
            return

    # Validate current environment
    print("\n📋 Validating environment setup...")
    validation = validate_environment_setup()

    for check, result in validation["checks"].items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}: {result}")

    if validation["suggestions"]:
        print("\n💡 Suggestions:")
        for suggestion in validation["suggestions"]:
            print(f"  - {suggestion}")

    # Create .env file
    print("\n📝 Creating .env file from template...")
    create_env_file_from_template()

    # Show next steps
    print("\n" + "=" * 60)
    print("✅ Configuration setup complete!")
    print("\n📋 Next steps:")
    print("  1. Edit the .env file and add your API keys:")
    print("     - OPENAI_API_KEY=your_key_here")
    print("     - GROQ_API_KEY=your_key_here")
    print("  2. Update any other configuration values as needed")
    print("  3. Run 'python main.py' to start the server")
    print("=" * 60)


if __name__ == "__main__":
    main()
