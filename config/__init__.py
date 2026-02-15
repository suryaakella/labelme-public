from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "forgeindex"
    postgres_password: str = "forgeindex_secret"
    postgres_db: str = "forgeindex"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "forgeindex"
    minio_secure: bool = False

    # Apify
    apify_api_token: str = ""

    # GDPR / LLM
    anthropic_api_key: str = ""
    google_api_key: str = ""
    gdpr_query_check_enabled: bool = True
    gdpr_pipeline_check_enabled: bool = True
    gdpr_llm_model: str = "gemini-2.0-flash-lite"
    annotation_llm_model: str = "gemini-2.0-flash"
    annotation_max_output_tokens: int = 4096

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ML Models
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    yolo_model: str = "yolov8n.pt"
    whisper_model: str = "base"

    # yt-dlp cookies (needed for Instagram downloads)
    ytdlp_cookies_file: str = ""

    # Pipeline
    max_workers: int = 4
    keyframe_interval: int = 30
    scene_threshold: float = 0.3
    max_keyframes: int = 100
    dedup_threshold: float = 0.95
    yolo_confidence: float = 0.3
    yolo_batch_size: int = 16
    min_activity_duration: float = 2.0
    video_max_resolution: int = 720

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
