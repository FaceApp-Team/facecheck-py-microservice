import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
