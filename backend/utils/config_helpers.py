"""
Configuration utility functions for Prompt Optimizer Backend

This module provides helper functions for configuration management,
validation, and environment setup.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger()


def load_env_file(env_file: str) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    env_vars = {}

    if not Path(env_file).exists():
        logger.warning(f"Environment file not found: {env_file}")
        return env_vars

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE format
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value

    return env_vars


def setup_environment_from_config(environment: str = "development") -> None:
    """Load environment variables from config directory."""
    config_dir = Path(__file__).parent.parent.parent / "config"
    env_config_file = config_dir / environment / "config.env"

    if env_config_file.exists():
        env_vars = load_env_file(str(env_config_file))
        # Set environment variables
        for key, value in env_vars.items():
            if key not in os.environ:  # Don't override existing env vars
                os.environ[key] = value
        logger.info(f"Loaded environment configuration from {env_config_file}")
    else:
        logger.warning(f"Environment config file not found: {env_config_file}")


def check_database_connection(database_url: str) -> bool:
    """Check if database is accessible."""
    # This is a placeholder - will be implemented in Phase 1
    logger.debug(f"Checking database connection: {database_url}")
    return True


def check_redis_connection(redis_url: str) -> bool:
    """Check if Redis is accessible."""
    try:
        import redis

        client = redis.from_url(redis_url, socket_connect_timeout=2)
        client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


def check_cpp_module(cpp_module_path: str) -> bool:
    """Check if C++ module is available."""
    return Path(cpp_module_path).exists()


def validate_environment_setup() -> Dict[str, Any]:
    """Comprehensive environment setup validation."""
    results = {
        "valid": True,
        "checks": {},
        "missing_dependencies": [],
        "suggestions": [],
    }

    # Check Python version
    python_version = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
    results["checks"]["python_version"] = python_version
    if os.sys.version_info < (3, 8):
        results["suggestions"].append("Python 3.8+ required")

    # Check for virtual environment
    in_venv = hasattr(os.sys, "real_prefix") or (
        hasattr(os.sys, "base_prefix") and os.sys.base_prefix != os.sys.prefix
    )
    results["checks"]["virtual_environment"] = in_venv
    if not in_venv:
        results["suggestions"].append("Consider using a virtual environment")

    # Check for .env file
    env_file_exists = Path(".env").exists()
    results["checks"][".env_file"] = env_file_exists
    if not env_file_exists:
        results["suggestions"].append("Create a .env file from config/env.example")

    # Check required directories
    required_dirs = ["backend", "tests", "docs"]
    for dir_name in required_dirs:
        dir_exists = Path(dir_name).exists()
        results["checks"][f"dir_{dir_name}"] = dir_exists
        if not dir_exists:
            results["valid"] = False
            results["missing_dependencies"].append(f"Directory {dir_name}/")

    return results


def print_configuration_summary(settings) -> None:
    """Print a summary of the current configuration (safe for logging)."""
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    print(f"\nEnvironment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(
        f"Database: {settings.DATABASE_URL[:50]}..."
        if len(settings.DATABASE_URL) > 50
        else f"Database: {settings.DATABASE_URL}"
    )
    print(f"Redis Enabled: {settings.REDIS_ENABLED}")
    print(f"Cache Enabled: {settings.CACHE_ENABLED}")
    print(f"C++ Module: {'Enabled' if settings.CPP_MODULE_ENABLED else 'Disabled'}")

    # Check API keys (don't print values)
    api_keys_status = settings.validate_api_keys()
    if api_keys_status:
        print(f"\n⚠️  Missing API keys: {', '.join(api_keys_status)}")
    else:
        print("\n✅ All API keys are set")

    print("=" * 60 + "\n")


def create_env_file_from_template(
    template_file: str = "config/env.example", output_file: str = ".env"
) -> None:
    """Create a .env file from a template."""
    template_path = Path(template_file)
    output_path = Path(output_file)

    if not template_path.exists():
        logger.error(f"Template file not found: {template_file}")
        return

    if output_path.exists():
        logger.warning(f"Output file already exists: {output_file}")
        return

    # Copy template to .env
    with open(template_path, "r") as src, open(output_path, "w") as dst:
        dst.write(src.read())

    logger.info(f"Created {output_file} from {template_file}")
    logger.info(f"Please edit {output_file} and add your API keys and configuration")
