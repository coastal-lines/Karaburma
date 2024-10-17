import os
import pytest
from httpx import AsyncClient
from typing import AsyncGenerator
from starlette.testclient import TestClient

from karaburma.api.controllers.root_controller import root_router
from karaburma.api.controllers.base64_controller import Base64Controller
from karaburma.api.controllers.file_controller import FileController
from karaburma.api.controllers.screenshot_controller import ScreenshotController
from karaburma.api.services.base64_service import Base64Service
from karaburma.api.services.file_service import FileService
from karaburma.api.services.screenshot_service import ScreenshotService
from karaburma.api.karaburma_api import KaraburmaApiService
from karaburma.utils import files_helper
from karaburma.main import Karaburma


HOST = "127.0.0.1"
PORT = 8900

@pytest.fixture(autouse=True, scope='session')
async def prepare_api_service():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = Karaburma(config_path, "file", "default", False)

    screenshot_service = ScreenshotService(karaburma)
    screenshot_controller = ScreenshotController(screenshot_service)

    file_service = FileService(karaburma)
    file_controller = FileController(file_service)

    base64image_service = Base64Service(karaburma)
    base64image_controller = Base64Controller(base64image_service)

    karaburma_service = KaraburmaApiService(HOST, PORT, config_path, "file", "default", False)
    karaburma_service.get_app().include_router(root_router)
    karaburma_service.get_app().include_router(screenshot_controller.get_screenshot_router())
    karaburma_service.get_app().include_router(file_controller.get_file_router())
    karaburma_service.get_app().include_router(base64image_controller.get_base64image_router())

    yield karaburma_service

@pytest.fixture(scope="session")
async def testclient(prepare_api_service) -> AsyncGenerator[TestClient, None]:
    async with AsyncClient(app=prepare_api_service.get_app()) as testclient:
        yield testclient
        await prepare_api_service.stop_karaburma_service()