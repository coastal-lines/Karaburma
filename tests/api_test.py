import asyncio
import json
import httpx
import pytest
from starlette.testclient import TestClient

from karaburma.api.models.response_model import RootKaraburmaResponse
from tests.conftest import setup_karaburma_api_file_mode



@pytest.mark.asyncio
async def test_200_OK_server_availability(setup_karaburma_api_file_mode):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8900/")

    assert response.status_code == 200
    assert response.json() == {'message': 'Uvicorn server was started for Karaburma.'}

@pytest.mark.asyncio
async def test_file_contains_any_button(setup_karaburma_api_file_mode):
    url = "http://127.0.0.1:8900/api/v1/file/"

    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": img_path,
        "type_element": "all"
    }

    client = TestClient(setup_karaburma_api_file_mode)
    response = client.post("/api/v1/file/", headers=headers, json=payload)

    data = json.loads(response.text)
    root_object = RootKaraburmaResponse(**data)

    #assert response.status_code == 200
    #assert response.json() == {'message': 'Uvicorn server was started for Karaburma.'}

@pytest.mark.asyncio
async def test_file_contains_any_button____(setup_karaburma_api_file_mode):
    url = "http://127.0.0.1:8900/api/v1/file/"

    payload_ = {
        "image_file_path": img_path,
        "type_element": "all"
    }


    payload = json.dumps({
        "image_file_path": img_path,
        "type_element": "all"
    })

    headers = {
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("http://127.0.0.1:8900/api/v1/file/", json=payload_, headers=headers)
        await asyncio.sleep(3)
        print(response)
        #assert response.status_code == 200