import unittest
import os
import sys
from misc import get_config, tester
import os    
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))            
from server.src.server_lib import save_intial_model

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

def create_train_test_for_six_clients():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '6')
            save_intial_model(config)

        def test_intermediate_client_connections_module(self):
            config = get_config('test_scalability', '6')
            tester(config,6)

    return TrainerTest

def create_train_test_for_eight_clients():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '8')
            save_intial_model(config)

        def test_intermediate_client_connections_module(self):
            config = get_config('test_scalability', '8')
            tester(config,8)

    return TrainerTest

def create_train_test_for_ten_clients():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '10')
            save_intial_model(config)

        def test_intermediate_client_connections_module(self):
            config = get_config('test_scalability', '10')
            tester(config,10)

    return TrainerTest

class TestTrainer_2(create_train_test_for_two_clients()):
    'Test case for two clients'

class TestTrainer_4(create_train_test_for_four_clients()):
    'Test case for four clients'

class TestTrainer_6(create_train_test_for_six_clients()):
    'Test case for six clients'

class TestTrainer_8(create_train_test_for_eight_clients()):
    'Test case for eight clients'

class TestTrainer_10(create_train_test_for_ten_clients()):
    'Test case for ten clients'

if __name__ == '__main__':

    unittest.main()
