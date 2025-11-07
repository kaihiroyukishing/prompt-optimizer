"""
Configuration management for Prompt Optimizer Backend

This module handles all configuration settings and environment variables.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from pydantic import validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings configuration."""

    DEBUG: bool = False
    SECRET_KEY: str = "your-secret-key-change-in-production"
    API_V1_STR: str = "/api/v1"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    DATABASE_URL: str = "sqlite:///./prompt_optimizer.db"
    DATABASE_ECHO: bool = False

    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    REDIS_URL: str = "redis://localhost:6379"
    REDIS_ENABLED: bool = False

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    MAX_SIMILAR_PROMPTS: int = 5

    CPP_MODULE_ENABLED: bool = True
    CPP_MODULE_PATH: str = "./cpp/build/similarity.so"

    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_ENABLED: bool = True

    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_cors_origins(cls, v):
        """Assemble CORS origins from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        """Warn if using default secret key."""
        if v == "your-secret-key-change-in-production":
            logger.warning("Using default secret key! Change this in production.")
        return v

    def validate_api_keys(self) -> List[str]:
        """Validate that required API keys are set. Returns list of missing keys."""
        missing = []
        if not self.OPENAI_API_KEY or self.OPENAI_API_KEY == "your_openai_api_key_here":
            missing.append("OPENAI_API_KEY")
        if not self.GROQ_API_KEY or self.GROQ_API_KEY == "your_groq_api_key_here":
            missing.append("GROQ_API_KEY")
        return missing

    def validate_file_paths(self) -> List[str]:
        """Validate that file paths exist. Returns list of missing paths."""
        missing = []
        if self.CPP_MODULE_ENABLED:
            if not Path(self.CPP_MODULE_PATH).exists():
                missing.append(f"CPP_MODULE_PATH: {self.CPP_MODULE_PATH}")
        return missing

    def validate_configuration(self) -> dict:
        """Comprehensive validation of configuration. Returns validation results."""
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "missing_keys": [],
            "missing_files": [],
        }

        # Check for missing API keys
        missing_keys = self.validate_api_keys()
        if missing_keys:
            results["missing_keys"] = missing_keys
            results["warnings"].append(f"Missing API keys: {', '.join(missing_keys)}")

        # Check for missing file paths
        missing_files = self.validate_file_paths()
        if missing_files:
            results["missing_files"] = missing_files
            results["warnings"].append(f"Missing files: {', '.join(missing_files)}")

        # Basic validations
        if self.DEBUG and self.SECRET_KEY == "your-secret-key-change-in-production":
            results["warnings"].append("Using default secret key in debug mode")

        return results

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = True


settings = Settings()
