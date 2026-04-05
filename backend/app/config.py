"""
PlantHealth AI Backend - Configuration
=======================================
Settings management using Pydantic Settings.
"""

from typing import List, Optional
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Settings
    app_name: str = "PlantHealth AI"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Model Settings
    model_path: str = "models/plant_disease_model.pth"
    class_names_path: str = "models/class_names.json"
    image_size: int = 224
    device: str = "auto"  # auto, cuda, cpu
    
    # Inference Settings
    confidence_threshold: float = 0.1
    top_k: int = 5
    
    @property
    def model_file(self) -> Path:
        """Get absolute path to model file."""
        base_path = Path(__file__).parent.parent
        return base_path / self.model_path
    
    @property
    def class_names_file(self) -> Path:
        """Get absolute path to class names file."""
        base_path = Path(__file__).parent.parent
        return base_path / self.class_names_path


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
