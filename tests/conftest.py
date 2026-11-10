import pytest

from jarvis.config import Config
from jarvis.skills.base import SkillContext


@pytest.fixture
def config(tmp_path):
    """A config whose workspace and data directory live in a temp folder."""
    return Config(
        {
            "assistant": {"workspace": str(tmp_path / "workspace"), "data_dir": str(tmp_path / "data")},
            "llm": {"active": "mock", "profiles": {"mock": {"provider": "mock", "model": "mock"}}},
            "skills": {"plugin_dirs": []},
            "agents": {"dirs": []},
            "publish": {"output_dir": str(tmp_path / "workspace" / "published")},
            "trading": {"data_source": "synthetic"},
        }
    )


@pytest.fixture
def ctx(config):
    return SkillContext(config=config)
