import datetime
import json
import os
import time

import pytest
from starlette.testclient import TestClient

from karaburma.utils import files_helper


HOST = "127.0.0.1"
PORT = 8900
BASE_URL=f"http://{HOST}:{PORT}"
ALL_ELEMENTS_TEST_SCREEN = "test_images\\all_elements.png"

@pytest.mark.asyncio
async def test_element(testclient: TestClient):
    response = await testclient.get(f"{BASE_URL}/api/v1/")
    print(response.text)
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_file_contains_any_button(testclient: TestClient):
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": os.path.abspath(os.path.join(files_helper.get_tests_root_path(), ALL_ELEMENTS_TEST_SCREEN)),
        "type_element": "all",
        "is_read_text": False
    }

    response = await testclient.post(f"{BASE_URL}/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" for item in response_dict["elements"])

@pytest.mark.asyncio
async def test_file_contains_removetest_button(testclient: TestClient):
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        "image_file_path": os.path.abspath(os.path.join(files_helper.get_tests_root_path(), ALL_ELEMENTS_TEST_SCREEN)),
        "type_element": "all",
        "is_read_text": True
    }

    response = await testclient.post(f"{BASE_URL}/api/v1/file/", headers=headers, json=payload)

    response_dict = json.loads(response.text)
    assert any(item["label"] == "button" and item['text'] == "RemoveTest" for item in response_dict["elements"])





