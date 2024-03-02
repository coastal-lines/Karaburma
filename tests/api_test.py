import time
from starlette.testclient import TestClient
from tests.conftest import karaburma_api_service_file_mode


def test_200_OK_server_availability(karaburma_api_service_file_mode):
    time.sleep(5)
    client = TestClient(karaburma_api_service_file_mode)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Uvicorn server was started for Karaburma.'}