import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient

from karaburma.api.karaburma_api import KaraburmaApiService
from karaburma.utils import files_helper


@pytest_asyncio.fixture
async def setup_karaburma_api_file_mode():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = KaraburmaApiService("127.0.0.1", 8900, config_path, "file", "default", False)

    karaburma.start_karaburma_service_for_test_fixtures()
    await asyncio.sleep(5)

    yield karaburma.get_app()

    await karaburma.stop_karaburma_service()

@pytest.fixture(autouse=True, scope='session')
async def prepare_database():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = KaraburmaApiService("127.0.0.1", 8900, config_path, "file", "default", False)
    karaburma.start_karaburma_service_for_test_fixtures()
    await asyncio.sleep(2)
    yield karaburma

@pytest.fixture(scope="session")
async def ac(prepare_database) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=prepare_database.get_app(), base_url="http://127.0.0.1:8900") as ac:
        yield ac
        await prepare_database.stop_karaburma_service()
