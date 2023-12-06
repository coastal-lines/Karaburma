import os

from tests.conftest import karaburma_api_service
from utils import files_helper


def test_server_availability(karaburma_api_service):
    server_is_available = karaburma_api_service.check_server_availability()
    assert (server_is_available, "'Karaburma' as API service is not available")

def test_test():
    project_directory = files_helper.get_project_root_path()
    assert True