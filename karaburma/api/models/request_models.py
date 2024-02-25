from pydantic import BaseModel

class ScreenshotCustomElementRequest(BaseModel):
    type_element: str

class ScreenshotElementRequest(BaseModel):
    type_element: str
    is_fully_expanded: bool

class FileElementRequest(BaseModel):
    image_file_path: str
    type_element: str