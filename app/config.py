from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    agent_mode: Literal["demo", "live"] = "demo"
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    max_tool_rounds: int = Field(default=4, ge=1, le=8)
    max_tool_calls: int = Field(default=8, ge=1, le=16)
    request_timeout_seconds: float = Field(default=60, gt=0, le=120)
    llm_timeout_seconds: float = Field(default=20, gt=0, le=60)
    data_dir: Path = Path(".local")
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    retrieval_candidate_k: int = Field(default=10, ge=1, le=100)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)

    @field_validator("openai_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        url = urlsplit(value)
        local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("API 地址必须是无凭据、无查询参数的基础 URL")
        if url.scheme != "https" and not local_http:
            raise ValueError("远程大模型接口必须使用 HTTPS")
        return value.rstrip("/")
