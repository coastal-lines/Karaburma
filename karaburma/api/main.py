import time
import uvicorn
import requests
from fastapi import FastAPI, status

from karaburma.main import Karaburma


class KaraburmaApiService:
    def __init__(self, host, port, config_path, source_mode, detection_mode, logging):

        self.__server = None

        self._karaburma_instance = Karaburma(config_path, source_mode, detection_mode, logging)
        self.__app = FastAPI()

        self.__host = host
        self.__port = port

        @self.__app.get("/")
        def get_condition():
            return status.HTTP_200_OK

        @self.__app.get("/get_all_elements")
        def get_all_elements(img_path):
            return {"message": self._karaburma_instance.find_all_elements(img_path)}

    async def start_lena_service(self):
        #uvicorn.run(self.__app, host=self.__host, port=self.__port, reload=False, workers=1)

        self.__server = uvicorn.Server(self.__app, host=self.__host, port=self.__port, reload=False, workers=1)

        await self.__server.serve()

        #self.__server.should_exit = True

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

    def stop_lena_service(self):
        self.__server.should_exit = True
        self.__server.close()
