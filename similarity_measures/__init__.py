import os
import config_utils

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../configs')

def make(id, **kwargs):
    """
    Instantiate a python object from a config file.
    Args:
        id: path to the config file and key to instantiate
        kwargs: keyword arguments passed to the object constructor
    """
    return config_utils.make(id, config_dir=CONFIG_DIR, **kwargs)

