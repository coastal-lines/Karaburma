import asyncio
import os
import threading
import time
import pytest
from starlette.testclient import TestClient

from karaburma.api.karaburma_api import KaraburmaApiService
from utils import files_helper


@pytest.fixture
def karaburma_api_service_file_mode():
    source_mode = "file"
    detection_mode = "default"
    logging = False
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = KaraburmaApiService("127.0.0.1", 8900, config_path, source_mode, detection_mode, logging)

    yield karaburma.get_app()

    karaburma.stop_karaburma_service()

