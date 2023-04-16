import unittest
import os
import sys
from get_config import get_config
import multiprocessing
import os    
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            
from server.src.server import server_start  
import time    
                                                             
# This block of code enables us to call the script from command line.                                                                                
def execute(process):
    x,y = process.split(':')
    if x =='server':
        config = get_config('test_datasets', y)
        server_start(config)
    else:
        ##sleep 10 seconds to allow server to start
        time.sleep(60)
        os.system(f'{y}')                                                                                                 
 
def create_train_test_for_datasets():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavg')
            cls.config = config

        def test_MNIST(self):
            all_processes = ('server:MNIST', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_FashionMNIST(self):
            all_processes = ('server:FashionMNIST', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_CIFAR10(self):
            all_processes = ('server:CIFAR10', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_CIFAR100(self):
            all_processes = ('server:CIFAR100', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

class TestTrainer(create_train_test_for_datasets()):
    'Test case for all datasets'

if __name__ == '__main__':

    unittest.main()