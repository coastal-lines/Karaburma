from abc import ABC, abstractmethod


class BaseFindActions(ABC):
    def get_source_image(self):
        pass

    @abstractmethod
    def find_all_elements(self, screenshot):
        pass

    @abstractmethod
    def find_element(self, screenshot, element_type):
        pass