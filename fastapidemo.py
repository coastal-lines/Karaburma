import base64
import io

import cv2
import numpy as np
import uvicorn
import asyncio
from fastapi import FastAPI, UploadFile, Depends
from imageio import imread
from matplotlib import pyplot as plt

from pydantic import BaseModel


class ImageParams(BaseModel):
    image_base64: str
    param1: str
    param2: str

class ApiDemoFastApi:

    def __init__(self):
        self.app = FastAPI()

        #router/endpoint
        @self.app.get("/")
        async def root():
            return {"message": "Uvicorn server was started for Karaburma."}

        @self.app.post("/uploadfile/")
        def create_upload_file(image_params: ImageParams):
            image_base64 = image_params.image_base64
            param1 = image_params.param1
            param2 = image_params.param2

            print(image_base64)
            image_base64 = image_base64.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite("test.png", img)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return {"image_filename": "image.png", "param1": param1, "param2": param2}

    def startapi(self):
        uvicorn_config = uvicorn.Config(app=self.app, host="127.0.0.1", port=8900, reload=True, workers=1)
        server = uvicorn.Server(uvicorn_config)
        asyncio.run(server.serve())

'''
encoded_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAADCAIAAADUVFKvAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAO0lEQVQIHQEwAM//Adja3AYFAwYFBQIDAwD/AAHz8/T7+/sICAcCAgL///8ECAgHAAD//v7+AQICAQEBnsAQt6Elsf8AAAAASUVORK5CYII=".split(',')[1]
nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

a = ApiDemoFastApi()
a.startapi()