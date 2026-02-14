"""Configuration management for AIPlayground backend."""

from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    runpod_api_key: str = ""
    runpod_enabled: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Set RunPod API key globally for Flash
if settings.runpod_api_key:
    os.environ["RUNPOD_API_KEY"] = settings.runpod_api_key
