"""
Configuration Management for Health Screening Pipeline

Environment-based configuration using Pydantic Settings.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env file
    )
    
    # Application
    app_name: str = "Health Screening Pipeline"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]
    
    # Authentication
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="JWT secret key")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # LLM Configuration
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face API token")
    gemini_model: str = "gemini-1.5-flash-latest"
    # Medical AI models for validation and reports
    medical_model_1: str = "Intelligent-Internet/II-Medical-8B-1706:featherless-ai"  # User requested for validation
    medical_model_2: str = "openai/gpt-oss-120b:groq"  # GPT-OSS-120B via Groq for arbitration
    use_mock_llm: bool = Field(default=True, description="Use mock LLM for testing")
    
    # Data Ingestion
    video_frame_rate: int = 30
    sync_tolerance_ms: float = 50.0
    buffer_size: int = 100
    
    # RIS Simulation
    ris_sample_rate: int = 1000  # Hz
    ris_num_channels: int = 16
    
    # Paths
    data_dir: str = "data"
    reports_dir: str = "reports"
    upload_dir: str = "uploads"
    
    # Database
    database_url: str = "sqlite:///./health_screening.db"

    # Validation & Reliability Configuration
    enable_validation: bool = Field(default=True, description="Master switch for all validation logic")
    enable_plausibility: bool = Field(default=True, description="Enable physiological plausibility checks")
    enable_consistency: bool = Field(default=True, description="Enable cross-system consistency checks")
    enable_trust_envelope: bool = Field(default=True, description="Enable trust envelope aggregation")
    
    # Thresholds
    min_trust_reliability: float = Field(default=0.5, description="Reject if overall reliability is below this")
    confidence_penalty_threshold: float = Field(default=0.2, description="Penalty intensity for low plausibility")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
