import json
import os
import httpx
import pytest
from starlette.testclient import TestClient

from tests.conftest import setup_karaburma_api_file_mode
from karaburma.utils import files_helper


ALL_ELEMENTS_TEST_SCREEN = "test_images\\all_elements.png"

@pytest.mark.asyncio
async def test_200_OK_server_availability(setup_karaburma_api_file_mode):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8900/")

    assert response.status_code == 200
    assert response.json() == {'message': 'Uvicorn server was started for Karaburma.'}

@pytest.mark.asyncio
async def test_file_contains_any_button(setup_karaburma_api_file_mode):
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": os.path.abspath(os.path.join(files_helper.get_tests_root_path(), ALL_ELEMENTS_TEST_SCREEN)),
        "type_element": "all"
    }

    client = TestClient(setup_karaburma_api_file_mode)
    response = client.post("/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" for item in response_dict["elements"])
