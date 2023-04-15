import os
import json

def get_config(action, action2, config_path=""):

    root_path = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'configs')
    action = action + '.json'
    with open(os.path.join(config_path, action)) as f1:
        config = json.load(f1)
        config = config[action2]

    return config