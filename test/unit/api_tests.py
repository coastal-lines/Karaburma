import threading
import time
import pytest

from main import LenaApiService


@pytest.fixture
def api_service():
    api_service = LenaApiService("127.0.0.1", 8000, "config.json", "file", "default", False)

    # start service in the thread
    api_thread = threading.Thread(target=api_service.start_lena_service)
    api_thread.start()

    time.sleep(2)

    yield api_service

def test_server_availability(api_service):
    server_is_available = api_service.check_server_availability()

    assert server_is_available, "Lena as API service is not available"
