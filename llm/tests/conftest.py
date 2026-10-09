import os

os.environ.update({"ADVANCERAG_ENV_FILE": "none", "ENVIRONMENT": "test", "INTERNAL_SERVICE_TOKEN": "", "FREELLMAPI_API_KEY": "", "LOG_LEVEL": "WARNING"})

import httpx  # noqa: E402
import pytest  # noqa: E402

from mock_openai import VALID_KEY, ServerThread, app as mock_app, free_port, scenario  # noqa: E402


@pytest.fixture(scope="session")
def mock_server():
    port = free_port()
    with ServerThread(mock_app, port):
        yield f"http://127.0.0.1:{port}/v1"


@pytest.fixture(autouse=True)
def reset_scenario():
    scenario.reset()
    yield


@pytest.fixture
def make_provider(mock_server):
    from llm_service.providers.freellmapi import FreeLLMAPIProvider

    created = []

    def _make(**kw):
        params = dict(base_url=mock_server, api_key=VALID_KEY, default_model="auto", timeout=5, connect_timeout=1, max_retries=2, backoff=0.01)
        params.update(kw)
        p = FreeLLMAPIProvider(**params)
        created.append(p)
        return p

    yield _make


@pytest.fixture
async def api(mock_server):
    """The LLM service app wired to the mock gateway."""
    from llm_service import main
    from llm_service.providers import get_provider
    from llm_service.providers.freellmapi import FreeLLMAPIProvider

    provider = FreeLLMAPIProvider(base_url=mock_server, api_key=VALID_KEY, default_model="auto", timeout=5, max_retries=1, backoff=0.01)
    main.app.dependency_overrides[get_provider] = lambda: provider
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://llm") as client:
        yield client
    main.app.dependency_overrides.clear()
    await provider.aclose()
