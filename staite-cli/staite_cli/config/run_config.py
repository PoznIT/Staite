"""Configuration model and loader for StAIte."""

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)


class AzureConfig(BaseModel):  # plain BaseModel, not BaseSettings
    endpoint: str = Field(..., min_length=1)
    model: str | None
    api_key: str | None = None

class AnthropicConfig(BaseModel):  # plain BaseModel, not BaseSettings
    api_key: str
    model: str = "claude-haiku-4-5-20251001"

class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b"
    timeout: int = 300

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Loads settings from staite.yml."""

    def get_field_value(self, field, field_name):
        return self._yaml_data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._yaml_data

    def __init__(self, settings_cls):
        super().__init__(settings_cls)
        path = Path("staite.yml")
        self._yaml_data = yaml.safe_load(path.read_text()) or {} if path.exists() else {}


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    azure: AzureConfig | None = None
    anthropic: AnthropicConfig | None = None
    ollama: OllamaConfig | None = None

    project_name: str = Field(..., min_length=1)
    include: list[str] = Field(..., min_length=1)
    exclude: list[str] = Field(default_factory=list)
    instructions: str = ""
    conventions: str = ""
    provider: Literal["anthropic", "azure", "ollama"] = "anthropic"

    output: Path = Path(".staite/STATE.json")
    cache: Path = Path(".staite/cache.json")
    synthesis_cache: Path = Path(".staite/synthesis.json")
    regen_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("include", "exclude", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("must be a list")
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"each pattern must be a string, got {type(item)}")
        return v

    @field_validator("include")
    @classmethod
    def include_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("include list must contain at least one pattern")
        return v

    @model_validator(mode="after")
    def validate_provider_config(self) -> "RunConfig":
        if self.provider == "azure" and self.azure is None:
            raise ValueError("'azure' section is required when provider is 'azure'")
        if self.provider == "ollama" and self.ollama is None:
            self.ollama = OllamaConfig()  # fall back to defaults
        if self.provider == "anthropic" and self.anthropic is None:
            raise ValueError("'anthropic' section is required when provider is 'anthropic'")
        self.output = Path(self.output)
        self.cache = Path(self.cache)
        self.synthesis_cache = Path(self.synthesis_cache)
        return self

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
    ):
        return (
            init_settings,       # highest priority
            env_settings,        # env vars (ANTHROPIC_API_KEY etc.) override yaml
            dotenv_settings,     # .env file
            YamlConfigSettingsSource(settings_cls),  # yaml is the base/default
            file_secret_settings,
        )