import asyncio
import os
import time
import uvicorn
import requests
from typing import List
from fastapi import FastAPI, status
from numpy import ndarray
from pydantic import BaseModel

from elements.objects.element import Element
from elements.objects.table.table_element import TableElement
from karaburma.main import Karaburma
from utils import files_helper


class RequestParams(BaseModel):
    image_base64: str
    image_file_path: str
    type_element: str

class KaraburmaApiService:
    def __init__(self, host, port, config_path, source_mode, detection_mode, logging):

        self._server = None

        self._karaburma_instance = Karaburma(config_path, source_mode, detection_mode, logging)
        self._app = FastAPI()

        self._host = host
        self._port = port

        #endpoint
        @self._app.get("/", status_code=status.HTTP_200_OK)
        async def root():
            return {"message": "Uvicorn server was started for Karaburma."}

        @self._app.post("/api/v1/file/", status_code=status.HTTP_200_OK)
        def create_upload_file(request_params: RequestParams):
            image_file_path = request_params.image_file_path
            result_json = self._karaburma_instance.find_all_elements(image_file_path)
            return result_json

    def start_karaburma_service(self):
        uvicorn_config = uvicorn.Config(app=self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(uvicorn_config)
        asyncio.run(self._server.serve())

    def check_server_availability(self):
        url = "http://127.0.0.1:8000"
        max_retries = 10
        retries = 0
        wait_time = 2

        while retries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                print("Server is ready to accept requests")
                return True
            except requests.RequestException as e:
                print(f"Attempt {retries + 1} failed: {e}")
                retries += 1
                time.sleep(wait_time)

        print("Server did not start within the expected time")
        return False

    def stop_karaburma_service(self):
        self._server.should_exit = True
        self._server.close()


config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
k = KaraburmaApiService("127.0.0.1", 8900, config_path, "file", "default", False)
k.start_karaburma_service()