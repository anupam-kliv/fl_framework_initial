import unittest
import os
import sys
from get_config import get_config
import multiprocessing
import os    
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            
from server.src.server import server_start  
from server.src.server_lib import save_intial_model
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
 
def create_train_test_for_MNIST():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'MNIST')
            save_intial_model(config)

        def test_MNIST(self):
            all_processes = ('server:MNIST', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_FashionMnist():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'FashionMNIST')
            save_intial_model(config)

        def test_FashionMnist(self):
            all_processes = ('server:FashionMNIST', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_CIFAR10():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'CIFAR10')
            save_intial_model(config)

        def test_FashionMnist(self):
            all_processes = ('server:CIFAR10', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_CIFAR100():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'CIFAR100')
            save_intial_model(config)

        def test_FashionMnist(self):
            all_processes = ('server:CIFAR100', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest


class TestTrainer1(create_train_test_for_MNIST()):
    'Test case for MNIST dataset'

class TestTrainer2(create_train_test_for_FashionMnist()):
    'Test case for FashionMNIST dataset'

class TestTrainer3(create_train_test_for_CIFAR10()):
    'Test case for CIFAR10 dataset'

class TestTrainer4(create_train_test_for_CIFAR100()):
    'Test case for CIFAR100 dataset'

if __name__ == '__main__':

    unittest.main()