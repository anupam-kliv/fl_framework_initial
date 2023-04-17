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
        config = get_config('test_modules', y)
        server_start(config)
    elif x == 'client':
        ##sleep 10 seconds to allow server to start
        time.sleep(10)
        os.system(f'{y}')  
    elif x == 'client2':
        time.sleep(12)
        os.system(f'{y}') 
    else:
        ##sleep 20 seconds to allow client to join after server has started training
        time.sleep(20)
        os.system(f'{y}')                                                                                              
 
def create_train_test_for_verification_module():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_modules', 'verification')
            save_intial_model(config)

        def test_verification_module(self):
            all_processes = ('server:verification', 'client:cd client ;python client.py', 'client2:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 3)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_timeout_module():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_modules', 'timeout')
            save_intial_model(config)

        def test_verification_module(self):
            all_processes = ('server:timeout', 'client:cd client ;python client.py', 'client2:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 3)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_intermediate_connection_module():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_modules', 'intermediate')
            save_intial_model(config)

        def test_verification_module(self):
            all_processes = ('server:intermediate', 'client:cd client ;python client.py', 'late_client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 3)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

class TestTrainer_verification(create_train_test_for_verification_module()):
    'Test case for verification module'

class TestTrainer_timeout(create_train_test_for_timeout_module()):
    'Test case for timeout module'

class TestTrainer_intermediate(create_train_test_for_intermediate_connection_module()):
    'Test case for intermediate client connections module'

if __name__ == '__main__':

    unittest.main()
