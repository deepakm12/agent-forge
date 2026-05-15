"""Application settings loaded from environment variables and an optional .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration; values are read from env vars and the project .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.0
    openai_timeout: int = 60
    max_retries: int = 3
    csv_max_size_mb: int = 50
    csv_max_rows: int = 500_000
    max_iterations: int = 2
    output_dir: str = "outputs"
    db_path: str = "data/sessions.db"
    log_level: str = "INFO"


settings = Settings()
