from pathlib import Path
from typing import List, ClassVar

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChromaDbSettings(BaseModel):
    host: str
    port: int

class OllamaSettings(BaseModel):
    url: str
    model: str


class ServerSettings(BaseModel):
    transport: str = Field(default="stdio")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    state: List[Path] = Field(default_factory=list)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    __config_file: ClassVar[Path]   # default

    chroma: ChromaDbSettings = Field(default_factory=ChromaDbSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    log_level: str = Field(default="INFO")
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_config_file(cls, config_file: Path | str, **kwargs) -> "AppSettings":
        cls.__config_file = Path(config_file)
        return cls(**kwargs)

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
            init_settings,
            YamlConfigSettingsSource(settings_cls, cls.__config_file),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def yaml_file_settings() -> dict:
    path = Path("config.dev.yml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}

from pydantic_settings import PydanticBaseSettingsSource

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls, config_path: Path):
        super().__init__(settings_cls)
        self.config_path = config_path

    def get_field_value(self, field, field_name):
        return None, field_name, False  # not used; we override __call__

    def __call__(self) -> dict:
        if not self.config_path.exists():
            return {}
        return yaml.safe_load(self.config_path.read_text()) or {}
