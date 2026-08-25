import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "ASRP DineX AI Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security / API Key
    AI_API_KEY: str = "asrp_ai_secret_key_2026"
    
    # Backend URL
    BACKEND_API_URL: str = "http://localhost:5100"
    
    # LLM Settings (OpenAI / Gemini / Local Fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Embedding Model Settings
    EMBEDDING_PROVIDER: str = "fallback"  # "openai" | "gemini" | "fallback"
    EMBEDDING_DIMENSION: int = 1536
    
    # Database / Vector Store
    POSTGRES_DB_URL: Optional[str] = None
    VECTOR_STORAGE_DIR: str = "./data/vectors"

settings = Settings()
