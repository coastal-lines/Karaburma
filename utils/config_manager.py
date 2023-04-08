from config import config_from_json


class ConfigManager:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.conf = config_from_json(config_path, read_from_file=True)
        return cls._instance

    @property
    def config(self):
        return self._instance.conf

    @config.setter
    def config(self, new_config):
        self._instance.conf = new_config