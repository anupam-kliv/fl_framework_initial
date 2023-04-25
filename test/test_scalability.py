import unittest
import os
import sys
from misc import get_config, tester
import multiprocessing
import os    
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            
from server.src.server_lib import save_intial_model
import time                                                                                             
 
def create_train_test_for_one_client():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '1')
            save_intial_model(config)

        def test_verification_module(self):
            config = get_config('test_scalability', '1')
            tester(config,1)

    return TrainerTest

def create_train_test_for_two_clients():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '2')
            save_intial_model(config)

        def test_timeout_module(self):
            config = get_config('test_scalability', '2')
            tester(config,2)

    return TrainerTest

def create_train_test_for_three_clients():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '3')
            save_intial_model(config)

        def test_intermediate_client_connections_module(self):
            config = get_config('test_scalability', '3')
            tester(config,3)

    return TrainerTest

def create_train_test_for_four_clients():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '4')
            save_intial_model(config)

        def test_intermediate_client_connections_module(self):
            config = get_config('test_scalability', '4')
            tester(config,4)

    return TrainerTest

class TestTrainer_1(create_train_test_for_one_client()):
    'Test case for one client'

class TestTrainer_2(create_train_test_for_two_clients()):
    'Test case for two clients'

class TestTrainer_3(create_train_test_for_three_clients()):
    'Test case for three clients'

class TestTrainer_4(create_train_test_for_four_clients()):
    'Test case for four clients'

if __name__ == '__main__':

    unittest.main()
