import functools
from collections import OrderedDict
import torch

#averages all of the given state dicts
class feddyn():

    def __init__(self, config):
        self.algorithm = "Mime"
        self.lr = 1.0
        self.momentum = 0.9
        self.h = None
        self.alpha = 0.01
    
    def aggregate(self,server_model_state_dict, state_dicts):
        
        keys = server_model_state_dict.keys() #List of keys in a state_dict   

        if (not(self.h)): #If self.h = None, then the following line will execute. So only at first round, it'll execute
            self.m = [torch.zeros_like(server_model_state_dict[key]) for key in server_model_state_dict.keys()]

        sum_y = OrderedDict() #This will be our new server_model_state_dict
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            sum_y[key] = current_key_sum
        
        delta_x = [torch.zeros_like(param) for param in server_model_state_dict]
        for d_x, s_y, x in zip(delta_x, sum_y, server_model_state_dict):
            d_x.data = s_y.data - x.data

        #Update h
        for h, d_x in zip(self.h, delta_x):
            h.data -= (self.alpha/len(state_dicts)) * d_x.data

        #Update x
        for x, s_y, h in zip(server_model_state_dict, sum_y, self.h):
            x.data = (s_y.data/len(state_dicts)) - (h.data/self.alpha)         

        return server_model_state_dict
    