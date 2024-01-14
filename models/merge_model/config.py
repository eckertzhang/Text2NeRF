import os
import yaml


class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            ## Adapt to the new environment where the yaml's version is higher.
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        return self._dict[name]