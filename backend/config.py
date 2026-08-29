import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application Settings loaded from environment variables or .env file."""
    
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Razorpay Test Mode Credentials & Toggle
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_ENABLED: bool = False
    
    # LLM Settings
    LLM_API_KEY: str = ""
    
    # Database
    DATABASE_URL: str = "sqlite:///./recover_ai.db"
    
    # Policy file path
    POLICY_FILE_PATH: str = os.path.join("policies", "recovery_policy.yaml")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
