import unittest
import os
# import subprocess
# from get_config import get_config

import multiprocessing
import os                                                               
                                                                        
# This block of code enables us to call the script from command line.                                                                                
def execute(process):                                                             
    os.system(f'{process}')                                                                                                 
 
def create_train_test_for_phase1():
    class TrainerTest(unittest.TestCase):
        @classmethod
        # def setUpClass(cls):
        #     config = get_config(action='train', phase=1)
        #     cls.config = config
        #     if not os.path.exists(config["image_path"]):
        #         download_data(phase=1)

        def test_trainer(self):
            all_processes = ('cd server ;python start_server.py', 'sleep 10s ;cd client ;python client.py')                                                                       
            process_pool = multiprocessing.Pool(processes = 2)                                                        
            process_pool.map(execute, all_processes)
            

    return TrainerTest

class TestTrainer(create_train_test_for_phase1()):
    'Test case for all algorithms'

if __name__ == '__main__':

    unittest.main()