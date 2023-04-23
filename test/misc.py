import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))        
from server.src.server import server_start 
import threading
import time
from torch.multiprocessing import Process
from torch import multiprocessing
def get_config(action, action2, config_path=""):

    root_path = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'configs')
    action = action + '.json'
    with open(os.path.join(config_path, action)) as f1:
        config = json.load(f1)
        config = config[action2]

    return config

def execute(process):
    os.system(f'{process}')    

def tester(config , no_of_clients):

    multiprocessing.set_start_method('spawn', force=True)
    server = Process(target=server_start, args=(config,))
    clients = []
    server.start()
    time.sleep(5)
    for i in range(no_of_clients):
        client = Process(target=execute, args=(f'cd client ;python client.py',))
        clients.append(client)
        client.start()
        time.sleep(2)
        
    for i in range(no_of_clients):
        clients[i].join()
    server.join()

        
