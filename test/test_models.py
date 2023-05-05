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
 
def create_train_test_for_LeNet():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'LeNet')
            save_intial_model(config)

        def test_LeNet(self):
            print("\n==========================LeNet Testing==========================")
            config = get_config('test_models', 'LeNet')
            tester(config,1)

    return TrainerTest

def create_train_test_for_resnet18():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'resnet18')
            save_intial_model(config)

        def test_resnet18(self):
            print("\n==========================Resnet18 Testing==========================")
            config = get_config('test_models', 'resnet18')
            tester(config,1)

    return TrainerTest

def create_train_test_for_resnet50(): 
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'resnet50')
            save_intial_model(config)

        def test_resnet18(self):
            print("\n==========================Resnet50 Testing==========================")
            config = get_config('test_models', 'resnet50')
            tester(config,1)

    return TrainerTest

def create_train_test_for_vgg16():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'vgg16')
            save_intial_model(config)

        def test_vgg16(self):
            print("\n==========================VGG 16 Testing==========================")
            config = get_config('test_models', 'vgg16')
            tester(config,1)

    return TrainerTest

def create_train_test_for_AlexNet():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'AlexNet')
            save_intial_model(config)

        def test_vgg16(self):
            print("\n==========================AlexNet Testing==========================")
            config = get_config('test_models', 'AlexNet')
            tester(config,1)

    return TrainerTest

class TestTrainer_LeNet(create_train_test_for_LeNet()):
    'Test case for LeNet model'

class TestTrainer_resnet18(create_train_test_for_resnet18()):
    'Test case for resnet18 model'

class TestTrainer_resnet50(create_train_test_for_resnet50()):
    'Test case for resnet50 model'

class TestTrainer_vgg16(create_train_test_for_vgg16()):
    'Test case for vgg16 model'

class TestTrainer_AlexNet(create_train_test_for_AlexNet()):
    'Test case for AlexNet model'

if __name__ == '__main__':

    unittest.main()
