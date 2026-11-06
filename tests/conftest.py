import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, agent_mode="demo", data_dir=tmp_path, openai_api_key="")
    with TestClient(create_app(settings)) as instance:
        yield instance


@pytest.fixture
def live_client(tmp_path):
    settings = Settings(
        _env_file=None,
        agent_mode="live",
        data_dir=tmp_path,
        openai_api_key="test-key-not-real",
        max_tool_rounds=2,
        max_tool_calls=3,
    )
    with TestClient(create_app(settings)) as instance:
        yield instance
