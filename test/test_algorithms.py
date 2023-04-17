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
                                                             
# This block of code enables us to call the server_start function and client script from command line.                                                                                
def execute(process):
    x,y = process.split(':')
    if x =='server':
        config = get_config('test_algorithms', y)
        server_start(config)
    else:
        ##sleep 10 seconds to allow server to start
        time.sleep(10)
        os.system(f'{y}')                                                                                                 

def create_train_test_for_fedavg():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavg')
            save_intial_model(config)

        def test_fedavg(self):
            all_processes = ('server:fedavg', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

def create_train_test_for_fedadagrad():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedadagrad')
            save_intial_model(config)

        def test_fedadagrad(self):
            all_processes = ('server:fedadagrad', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

def create_train_test_for_fedadam():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedadam')
            save_intial_model(config)
        
        def test_fedadam(self):
            all_processes = ('server:fedadam', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
        
    return TrainerTest

def create_train_test_for_fedavgm():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavgm')
            save_intial_model(config)

        def test_fedavgm(self):
            all_processes = ('server:fedavgm', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

def create_train_test_for_feddyn():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'feddyn')
            save_intial_model(config)

        def test_feddyn(self):
            all_processes = ('server:feddyn', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

def create_train_test_for_fedyogi():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedyogi')
            save_intial_model(config)
        
        def test_fedyogi(self):
            all_processes = ('server:fedyogi', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
    
    return TrainerTest

def create_train_test_for_mime():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'mime')
            save_intial_model(config)

        def test_mime(self):
            all_processes = ('server:mime', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

def create_train_test_for_mimelite():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'mimelite')
            save_intial_model(config)

        def test_mimelite(self):
            all_processes = ('server:mimelite', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

def create_train_test_for_scaffold():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'scaffold')
            save_intial_model(config)

        def test_scaffold(self):
            all_processes = ('server:scaffold', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest


class TestTrainer_fedavg(create_train_test_for_fedavg()):
    'Test case for fedavg'

class TestTrainer_fedadagrad(create_train_test_for_fedadagrad()):
    'Test case for fedadagrad'

class TestTrainer_fedadam(create_train_test_for_fedadam()):
    'Test case for fedadam'

class TestTrainer_fedavgm(create_train_test_for_fedavgm()):
    'Test case for fedavgm'

class TestTrainer_feddyn(create_train_test_for_feddyn()):
    'Test case for feddyn'

class TestTrainer_fedyogi(create_train_test_for_fedyogi()):
    'Test case for fedyogi'

class TestTrainer_mime(create_train_test_for_mime()):
    'Test case for mime'

class TestTrainer_mimelite(create_train_test_for_mimelite()):
    'Test case for mimelite'

class TestTrainer_scaffold(create_train_test_for_scaffold()):
    'Test case for scaffold'

if __name__ == '__main__':

    unittest.main()