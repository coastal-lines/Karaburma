import json
import os
import pytest
from httpx import AsyncClient

from karaburma.utils import files_helper


ALL_ELEMENTS_TEST_SCREEN = "test_images\\all_elements.png"

@pytest.mark.asyncio
async def test_200_OK_server_availability(ac: AsyncClient):
    response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Uvicorn server was started for Karaburma.'}

@pytest.mark.asyncio
async def test_file_contains_any_button(ac: AsyncClient):
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": os.path.abspath(os.path.join(files_helper.get_tests_root_path(), ALL_ELEMENTS_TEST_SCREEN)),
        "type_element": "all",
        "is_read_text": False
    }

    response = await ac.post("/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" for item in response_dict["elements"])

@pytest.mark.asyncio
async def test_file_contains_removetest_button(ac: AsyncClient):
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": os.path.abspath(os.path.join(files_helper.get_tests_root_path(), ALL_ELEMENTS_TEST_SCREEN)),
        "type_element": "all",
        "is_read_text": True
    }

    response = await ac.post("/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" and item['text'] == "RemoveTest" for item in response_dict["elements"])

'''
@pytest.mark.asyncio
async def test_200_OK_server_availability(ac: AsyncClient):
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
        "type_element": "all",
        "is_read_text": False
    }

    client = TestClient(setup_karaburma_api_file_mode)
    response = client.post("/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" for item in response_dict["elements"])

@pytest.mark.asyncio
async def test_file_contains_removetest_button(setup_karaburma_api_file_mode):
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": os.path.abspath(os.path.join(files_helper.get_tests_root_path(), ALL_ELEMENTS_TEST_SCREEN)),
        "type_element": "all",
        "is_read_text": True
    }

    client = TestClient(setup_karaburma_api_file_mode)
    response = client.post("/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" and item['text'] == "RemoveTest" for item in response_dict["elements"])
'''