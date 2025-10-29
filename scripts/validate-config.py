#!/usr/bin/env python3
"""
Configuration validation script for Prompt Optimizer

This script validates the current configuration and provides detailed feedback.
"""

import sys
from pathlib import Path

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from backend.app.core.config import get_settings
from backend.utils.config_helpers import (
    check_cpp_module,
    check_database_connection,
    check_redis_connection,
    print_configuration_summary,
)

# Configure logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger()


def main():
    print("🔍 Prompt Optimizer Configuration Validation")
    print("=" * 60)

    # Get settings
    settings = get_settings()

    # Print configuration summary
    print_configuration_summary(settings)

    # Validate configuration
    print("🔎 Running validation checks...\n")
    validation_results = settings.validate_configuration()

    # Display results
    if validation_results["errors"]:
        print("❌ Configuration Errors:")
        for error in validation_results["errors"]:
            print(f"   - {error}")

    if validation_results["warnings"]:
        print("\n⚠️  Configuration Warnings:")
        for warning in validation_results["warnings"]:
            print(f"   - {warning}")

    if validation_results["valid"] and not validation_results["warnings"]:
        print("✅ Configuration is valid!")

    # Test connections
    print("\n🌐 Testing connections...\n")

    # Database
    print("📊 Database:")
    db_status = check_database_connection(settings.DATABASE_URL)
    print(
        f"   {'✅' if db_status else '❌'} Connection {'OK' if db_status else 'FAILED'}"
    )

    # Redis
    if settings.REDIS_ENABLED:
        print("\n🔴 Redis:")
        redis_status = check_redis_connection(settings.REDIS_URL)
        print(
            f"   {'✅' if redis_status else '❌'} Connection {'OK' if redis_status else 'FAILED'}"
        )
    else:
        print("\n🔴 Redis: Disabled")

    # C++ Module
    if settings.CPP_MODULE_ENABLED:
        print("\n⚙️  C++ Module:")
        cpp_status = check_cpp_module(settings.CPP_MODULE_PATH)
        print(
            f"   {'✅' if cpp_status else '⚠️ '} Module {'Found' if cpp_status else 'Not Found'}"
        )
        if not cpp_status:
            print(f"      Path: {settings.CPP_MODULE_PATH}")
    else:
        print("\n⚙️  C++ Module: Disabled")

    print("\n" + "=" * 60)

    if validation_results["valid"]:
        print("✅ Configuration validation complete")
        sys.exit(0)
    else:
        print("❌ Configuration validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
