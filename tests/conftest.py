import os
import threading
import time
import pytest

from karaburma.api.main import KaraburmaApiService
from utils import files_helper


@pytest.fixture
def karaburma_api_service():
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    api_service = KaraburmaApiService("127.0.0.1", 8000, config_path, "file", "default", False)

    api_service.start_karaburma_service()

    time.sleep(2)

    yield api_service

    api_service.stop_karaburma_service()

@pytest.fixture
def karaburma_api_service_():
    config_path = os.path.dirname(os.path.realpath("__file__")).replace("\\", "/").replace("tests", "karaburma//config.json")
    print("\n -------------------------------")
    print(config_path)
    print("\n -------------------------------")

    api_service = KaraburmaApiService("127.0.0.1", 8000, config_path, "file", "default", False)

    api_thread = threading.Thread(target=api_service.start_karaburma_service())
    api_thread.start()

    time.sleep(2)

    yield api_service

    api_service.stop_karaburma_service()

    api_thread.join()