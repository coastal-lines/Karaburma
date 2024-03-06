import asyncio
import os
import pytest_asyncio

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