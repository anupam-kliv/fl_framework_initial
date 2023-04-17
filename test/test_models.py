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
        config = get_config('test_models', y)
        server_start(config)
    else:
        ##sleep 10 seconds to allow server to start
        time.sleep(60)
        os.system(f'{y}')                                                                                                 
 
def create_train_test_for_LeNet():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'LeNet')
            save_intial_model(config)

        def test_LeNet(self):
            all_processes = ('server:LeNet', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_resnet18():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'resnet18')
            save_intial_model(config)

        def test_resnet18(self):
            all_processes = ('server:resnet18', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_resnet50(): 
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'resnet50')
            save_intial_model(config)

        def test_resnet18(self):
            all_processes = ('server:resnet50', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_vgg16():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'vgg16')
            save_intial_model(config)

        def test_vgg16(self):
            all_processes = ('server:vgg16', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

    return TrainerTest

def create_train_test_for_AlexNet():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'AlexNet')
            save_intial_model(config)

        def test_vgg16(self):
            all_processes = ('server:AlexNet', 'client:cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)

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
