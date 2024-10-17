from config import config_from_json


class ConfigManager:
    _instance = None

    #def __new__(cls, config_path=None, source_mode=None, detection_mode=None, logging=None):
    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.conf = config_from_json(config_path, read_from_file=True)
            #cls._instance.src_mode = source_mode
            #cls._instance.dtc_mode = detection_mode
            #cls._instance.log = logging
        return cls._instance

    @property
    def config(self):
        return self._instance.conf

    @config.setter
    def config(self, new_config):
        self._instance.conf = new_config

    '''
        @property
        def logging(self):
            return self._instance.logging
    
        @logging.setter
        def logging(self, logging):
            self._instance.logging = logging
    
        @property
        def detection_mode(self):
            return self._instance.detection_mode
    
        @detection_mode.setter
        def detection_mode(self, detection_mode):
            self._instance.detection_mode = detection_mode
    
        @property
        def source_mode(self):
            return self._instance.source_mode
    
        @source_mode.setter
        def source_mode(self, source_mode):
            self._instance.source_mode = source_mode
    '''