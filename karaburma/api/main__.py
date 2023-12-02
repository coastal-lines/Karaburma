import sys

import uvicorn
from fastapi import FastAPI, APIRouter
from karaburma.main import Karaburma


class KaraburmaService:
    def __init__(self, config_path, source_mode, detection_mode, logging, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.config_path = config_path
        self.source_mode = source_mode
        self.detection_mode = detection_mode
        self.logging = logging

        #self.__NUMBER_PARAMETERS = 7

        #self.app = app
        #self.router = APIRouter()
        #self.app.include_router(self.router)
        #self.data = {}

        self.router = APIRouter()
        self.router.add_api_route("/elements/get_all", self.get_all_elements, methods=["GET"])

        self._karaburma = Karaburma(self.config_path, self.source_mode, self.detection_mode, self.logging)

        #@self.app.get("/elements/get_all")
        #def get_all_elements(file_path: str = None):
        #    return self._karaburma.find_all_elements(file_path)

    def get_all_elements(self, file_path: str = None):
        return self._karaburma.find_all_elements(file_path)

    '''
    def initialize_globals(self):
        karaburma = Karaburma(self.config_path, self.source_mode, self.detection_mode, self.logging)
        return karaburma
    '''

    def start(self):
        uvicorn.run("karaburma.api.main:app", host=self.host, port=self.port, reload=True)



'''
NUMBER_PARAMETERS = 7

app = FastAPI()
router = APIRouter()
app.include_router(router)
data = {}
'''

'''
@app.on_event('startup')
def init_data():
    data[1] = initialize_globals(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    return data
'''

'''
def initialize_globals(config_path, source_mode, detection_mode, logging):
    #config = ConfigManager(config_path)
    karaburma = Karaburma(config_path, source_mode, detection_mode, logging)
    return karaburma
'''
'''
if __name__ == "__main__":
    if(len(sys.argv) < NUMBER_PARAMETERS):
        raise IndexError("Some of parameters were not provided")

    karaburma = data[1]

    uvicorn.run("karaburma.api.main:app", host="0.0.0.0", port=8000, reload=True)
'''

'''
@app.get("/elements/get_all")
def get_all_elements(file_path: str = None):
    return data[1].find_all_elements(file_path)
'''

app = FastAPI()

if __name__ == "__main__":
    #app = FastAPI()
    l = KaraburmaService(r"Karaburma\karaburma\config.json", "file", "default", False)
    app.include_router(l.router)
    #uvicorn.run("karaburma.api.main:app", host="127.0.0.1", port=8000, reload=True)
    l.start()