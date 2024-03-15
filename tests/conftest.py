import asyncio
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient
from typing import AsyncGenerator

from karaburma.api.karaburma_api import KaraburmaApiService
from karaburma.utils import files_helper


HOST = "127.0.0.1"
PORT = 8900

@pytest_asyncio.fixture
async def setup_karaburma_api_file_mode():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = KaraburmaApiService(HOST, PORT, config_path, "file", "default", False)

    karaburma.start_karaburma_service_for_test_fixtures()
    await asyncio.sleep(5)

    yield karaburma.get_app()

    await karaburma.stop_karaburma_service()

@pytest.fixture(autouse=True, scope='session')
async def prepare_api_service():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = KaraburmaApiService(HOST, PORT, config_path, "file", "default", False)
    karaburma.start_karaburma_service_for_test_fixtures()
    await asyncio.sleep(2)
    yield karaburma

@pytest.fixture(scope="session")
async def ac(prepare_api_service) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=prepare_api_service.get_app(), base_url=f"http://{HOST}:{PORT}") as ac:
        yield ac
        await prepare_api_service.stop_karaburma_service()
