import unittest
from misc import get_config, tester
import torch.multiprocessing as multiprocessing
import os    
from server.src.server_lib import save_intial_model                                                                                            

def create_train_test_for_fedavg():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavg')
            save_intial_model(config['server'])

        def test_fedavg(self):
            print("\n==========================Fed Avg==========================")
            config = get_config('test_algorithms', 'fedavg')
            tester(config,2)
            
    return TrainerTest

def create_train_test_for_fedadagrad():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedadagrad')
            save_intial_model(config['server'])

        def test_fedadagrad(self):
            print("\n==========================Fed Adagrad==========================")
            config = get_config('test_algorithms', 'fedadagrad')
            tester(config,2)
            
    return TrainerTest

def create_train_test_for_fedadam():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedadam')
            save_intial_model(config['server'])
        
        def test_fedadam(self):
            print("\n==========================Fed Adam==========================")
            config = get_config('test_algorithms', 'fedadam')
            tester(config,2)
        
    return TrainerTest

def create_train_test_for_fedavgm():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedavgm')
            save_intial_model(config['server'])

        def test_fedavgm(self):
            print("\n==========================Fed Avgm==========================")
            config = get_config('test_algorithms', 'fedavgm')
            tester(config,2)
            
    return TrainerTest

def create_train_test_for_feddyn():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'feddyn')
            save_intial_model(config['server'])

        def test_feddyn(self):
            print("\n==========================Fed Dyn==========================")
            config = get_config('test_algorithms', 'feddyn')
            tester(config,2)
            
    return TrainerTest

def create_train_test_for_fedyogi():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'fedyogi')
            save_intial_model(config['server'])
        
        def test_fedyogi(self):
            print("\n==========================Fed Yogi==========================")
            config = get_config('test_algorithms', 'fedyogi')
            tester(config,2)
    
    return TrainerTest

def create_train_test_for_mime():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'mime')
            save_intial_model(config['server'])

        def test_mime(self):
            print("\n==========================Mime==========================")
            config = get_config('test_algorithms', 'mime')
            tester(config,2)
            
    return TrainerTest

def create_train_test_for_mimelite():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'mimelite')
            save_intial_model(config['server'])

        def test_mimelite(self):
            print("\n==========================MimeLite==========================")
            config = get_config('test_algorithms', 'mimelite')
            tester(config,2)
            
    return TrainerTest

def create_train_test_for_scaffold():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_algorithms', 'scaffold')
            save_intial_model(config['server'])

        def test_scaffold(self):
            print("\n==========================Scaffold==========================")
            config = get_config('test_algorithms', 'scaffold')
            tester(config,2)
            
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
