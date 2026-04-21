from enum import Enum
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMMode(str, Enum):
    MOCK = "mock"
    REAL = "real"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_mode: LLMMode = Field(default=LLMMode.MOCK, validation_alias="LLM_MODE")

    supabase_db_url: str = Field(validation_alias="SUPABASE_DB_URL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_ocr_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_OCR_MODEL")
    deepseek_api_key: str | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", validation_alias="DEEPSEEK_MODEL")

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")

    assets_dir: Path = Field(default=Path("./assets"), validation_alias="ASSETS_DIR")

    ocr_timeout_s: int = Field(default=30, validation_alias="OCR_TIMEOUT_S")
    llm_timeout_s: int = Field(default=20, validation_alias="LLM_TIMEOUT_S")
    tool_wall_timeout_s: int = Field(default=45, validation_alias="TOOL_WALL_TIMEOUT_S")
    max_file_size_mb: int = Field(default=10, validation_alias="MAX_FILE_SIZE_MB")
    max_files_per_run: int = Field(default=25, validation_alias="MAX_FILES_PER_RUN")

    allowed_extensions_raw: str = Field(
        default="jpg,jpeg,png,webp", validation_alias="ALLOWED_EXTENSIONS"
    )

    @property
    def allowed_extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions_raw.split(",") if e.strip()}

    @model_validator(mode="after")
    def _validate_real_mode_keys(self) -> "Settings":
        if self.llm_mode == LLMMode.REAL:
            missing = [
                name for name, val in (
                    ("OPENAI_API_KEY", self.openai_api_key),
                    ("DEEPSEEK_API_KEY", self.deepseek_api_key),
                ) if not val
            ]
            if missing:
                raise ValueError(
                    f"LLM_MODE=real requires: {', '.join(missing)}"
                )
        return self
