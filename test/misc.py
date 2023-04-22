import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))        
from server.src.server import server_start 
import threading
import time
import torch.multiprocessing as multiprocessing

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
    x, y = process.split(':')
    if x =='server':
        y1, y2 = y.split(',')
        config = get_config(y1,y2)
        server_start(config)
    else:
        y1, y2= y.split(',')
        time.sleep(y2)
        os.system(f'{y1}')    

def tester(config , no_of_clients = 1, t = 5):
    all_processes = ['server:'+config]
    print(all_processes)
    for _ in range(no_of_clients):
        # all_processes = all_processes+('client:cd client ;python client.py,'+str(t))
        all_processes.append('client:cd client ;python client.py,'+str(t))
        print(all_processes)
        t = t+2
    process_pool = multiprocessing.Pool(processes = no_of_clients + 1)
    process_pool.map(execute, all_processes)
   
        
