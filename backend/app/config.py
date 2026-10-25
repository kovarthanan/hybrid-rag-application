from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"

    max_query_chars: int = 500
    llm_timeout_s: float = 30.0

    data_dir: str = "data"
    kb_file: str = "data/spotify_web_app_architecture.pdf"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    relevance_threshold: float = 0.25   # min vector score to count as on-topic
    grounding_threshold: float = 0.5    # min share of answer words found in context
    enable_moderation: bool = True 

    rate_limit_per_min: int = 10     # per visitor
    rate_limit_per_day: int = 50     # per visitor
    global_limit_per_min: int = 60   # all visitors combined


@lru_cache
def get_settings() -> Settings:
    return Settings()