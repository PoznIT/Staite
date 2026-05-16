"""Configuration model and loader for StAIte."""

import logging
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class AzureConfig(BaseModel):
    """Azure AI Foundry connection settings."""

    endpoint: str = Field(..., min_length=1)
    """Azure AI Foundry endpoint URL.
    e.g. https://<resource>.services.ai.azure.com/models
    """

    api_key: str | None = None
    """Optional API key. If omitted, falls back to the AZURE_API_KEY environment
    variable, then DefaultAzureCredential (managed identity, Azure CLI, etc.).
    """


class StaiteConfig(BaseModel):
    """Full configuration for a StAIte run."""

    project_name: str = Field(..., min_length=1)
    include: list[str] = Field(..., min_length=1)
    exclude: list[str] = Field(default_factory=list)
    instructions: str = ""
    conventions: str = ""
    provider: Literal["anthropic", "azure"] = "anthropic"
    azure: AzureConfig | None = None
    model: str = "claude-haiku-4-5-20251001"
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
    def validate_provider_config(self) -> "StaiteConfig":
        """Ensure provider-specific config is present and paths are normalised."""
        if self.provider == "azure" and self.azure is None:
            raise ValueError("'azure' section is required when provider is 'azure'")
        self.output = Path(self.output)
        self.cache = Path(self.cache)
        self.synthesis_cache = Path(self.synthesis_cache)
        return self


def load_config(path: Path) -> StaiteConfig:
    """Load and validate a StAIte YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A validated StaiteConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    load_dotenv(path.parent / ".env")
    logger.debug("Loading config from %s", path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    config = StaiteConfig.model_validate(raw)
    logger.info(
        "Config loaded: project=%r, provider=%r, %d include pattern(s), %d exclude pattern(s)",
        config.project_name,
        config.provider,
        len(config.include),
        len(config.exclude),
    )
    return config
