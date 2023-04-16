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
        config = get_config('test_algorithms', y)
        server_start(config)
    else:
        ##sleep 10 seconds to allow server to start
        time.sleep(10)
        os.system(f'{y}')                                                                                                 
 
def create_train_test_for_algorithmss():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavg')
            cls.config = config

        def test_fedavg(self):
            # config = get_config('test_algorithms', 'fedavg')
            all_processes = ('server:fedavg', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)
        
        def test_fedadagrad(self):
            all_processes = ('server:fedadagrad', 'client:cd client ;python client.py')                                                                    
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_fedadam(self):
            all_processes = ('server:fedadam', 'client:cd client ;python client.py')                                                                      
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_fedavgm(self):
            all_processes = ('server:fedavgm', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_feddyn(self):
            all_processes = ('server:feddyn', 'client:cd client ;python client.py')                                                                      
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_fedyogi(self):
            all_processes = ('server:fedyogi', 'client:cd client ;python client.py')                                                                      
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_mime(self):
            all_processes = ('server:mime', 'client:cd client ;python client.py')                                                                      
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def test_mimelite(self):
            all_processes = ('server:mimelite', 'client:cd client ;python client.py')                                                                    
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

        def scaffold(self):
            all_processes = ('server:scaffold', 'client:cd client ;python client.py')
            process_pool = multiprocessing.Pool(processes = 2)
            process_pool.map(execute, all_processes)
            
    return TrainerTest

class TestTrainer(create_train_test_for_algorithmss()):
    'Test case for all algorithms'

if __name__ == '__main__':

    unittest.main()