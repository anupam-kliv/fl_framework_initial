import unittest
import os
import sys
from misc import get_config, tester
import multiprocessing
import os    
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            
from server.src.server import server_start 
from server.src.server_lib import save_intial_model
import time    
                                                             
def create_train_test_for_fedavg():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavg')
            save_intial_model(config)

        def test_fedavg(self):
            config = get_config('test_algorithms', 'fedavg',)
            tester(config, 2,  'test_algorithms.json')
            
    return TrainerTest

class TestTrainer_fedavg(create_train_test_for_fedavg()):
    'Test case for fedavg'

if __name__ == '__main__':

    unittest.main()