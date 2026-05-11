"""Tests for staite.config."""

from pathlib import Path

import pytest
import yaml

from staite.config import StaiteConfig, load_config


class TestStaiteConfig:
    def test_minimal_valid(self):
        cfg = StaiteConfig(project_name="Test", include=["src/"])
        assert cfg.project_name == "Test"
        assert cfg.exclude == []
        assert cfg.instructions == ""
        assert cfg.conventions == ""
        assert cfg.regen_threshold == 0.2
        assert cfg.model == "claude-haiku-4-5-20251001"

    def test_regen_threshold_bounds(self):
        cfg = StaiteConfig(project_name="X", include=["src/"], regen_threshold=0.0)
        assert cfg.regen_threshold == 0.0
        cfg2 = StaiteConfig(project_name="X", include=["src/"], regen_threshold=1.0)
        assert cfg2.regen_threshold == 1.0

    def test_regen_threshold_out_of_bounds(self):
        import pytest
        with pytest.raises(Exception):
            StaiteConfig(project_name="X", include=["src/"], regen_threshold=1.5)

    def test_full_valid(self):
        cfg = StaiteConfig(
            project_name="MyApp",
            include=["src/", "lib/"],
            exclude=["*.pyc", "__pycache__/"],
            instructions="A FastAPI app.",
            model="claude-opus-4-6",
            output=Path("out/STATE.xml"),
            cache=Path(".cache/cache.json"),
        )
        assert cfg.model == "claude-opus-4-6"
        assert len(cfg.include) == 2

    def test_empty_include_raises(self):
        with pytest.raises(ValueError, match="include"):
            StaiteConfig(project_name="X", include=[])

    def test_non_list_include_raises(self):
        with pytest.raises(ValueError):
            StaiteConfig(project_name="X", include="src/")  # type: ignore[arg-type]

    def test_non_string_in_include_raises(self):
        with pytest.raises(ValueError):
            StaiteConfig(project_name="X", include=[123])  # type: ignore[list-item]

    def test_paths_are_path_objects(self):
        cfg = StaiteConfig(project_name="X", include=["src/"], output="out.xml", cache="c.json")
        assert isinstance(cfg.output, Path)
        assert isinstance(cfg.cache, Path)


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path: Path):
        data = {"project_name": "Proj", "include": ["src/"]}
        cfg_file = tmp_path / "staite.yaml"
        cfg_file.write_text(yaml.dump(data))
        cfg = load_config(cfg_file)
        assert cfg.project_name == "Proj"

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="mapping"):
            load_config(cfg_file)

    def test_invalid_config_raises(self, tmp_path: Path):
        data = {"project_name": "X"}  # missing include
        cfg_file = tmp_path / "staite.yaml"
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(Exception):
            load_config(cfg_file)
