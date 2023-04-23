import unittest
import os
import sys
from misc import get_config, tester
import multiprocessing
import os    
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))             
from server.src.server_lib import save_intial_model
import time    
            
def create_train_test_for_MNIST():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'MNIST')
            save_intial_model(config)

        def test_MNIST(self):
            config = get_config('test_datasets', 'MNIST')
            tester(config,1)

    return TrainerTest

def create_train_test_for_FashionMnist():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'FashionMNIST')
            save_intial_model(config)

        def test_FashionMnist(self):
            config = get_config('test_datasets', 'FashionMNIST')
            tester(config,1)

    return TrainerTest

def create_train_test_for_CIFAR10():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'CIFAR10')
            save_intial_model(config)

        def test_CIFAR10(self):
            config = get_config('test_datasets', 'CIFAR10')
            tester(config,1)

    return TrainerTest

def create_train_test_for_CIFAR100():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_datasets', 'CIFAR100')
            save_intial_model(config)

        def test_CIFAR100(self):
            config = get_config('test_datasets', 'CIFAR100')
            tester(config,1)

    return TrainerTest


class TestTrainer_MNIST(create_train_test_for_MNIST()):
    'Test case for MNIST dataset'

class TestTrainer_FashionMNIST(create_train_test_for_FashionMnist()):
    'Test case for FashionMNIST dataset'

class TestTrainer_CIFAR10(create_train_test_for_CIFAR10()):
    'Test case for CIFAR10 dataset'

class TestTrainer_CIFAR100(create_train_test_for_CIFAR100()):
    'Test case for CIFAR100 dataset'

if __name__ == '__main__':

    unittest.main()