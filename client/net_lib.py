import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Grayscale, Compose

from get_data import get_data
from distribution import data_distribution
from data_utils import customDataset
from tqdm import tqdm
from copy import deepcopy
from math import ceil
import time
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(config):
    trainset, testset = get_data(config)
    data_distribution(config, trainset)
    data_path = os.path.join(os.getcwd(), 'Distribution/', config['dataset'], 'data_split_niid_'+ str(config['niid'])+'.pt')
    datasets = customDataset(config,  trainset, data_path)
    trainloader = DataLoader(datasets, batch_size= config['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=config['batch_size'])
    num_examples = {"trainset": len(datasets), "testset": len(testset)}
    return trainloader, testloader, num_examples

def flush_memory():
    torch.cuda.empty_cache()

def train_model(net, trainloader, epochs, deadline=None):
    #This function returns the difference between the trained model and the received model
    x = deepcopy(net)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break
               
    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data
    
    return net

def train_fedavg(net, trainloader, epochs, deadline=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break
    return net

def train_feddyn(net, trainloader, epochs, deadline=None):
    x = deepcopy(net) 
    prev_grads = None
    if os.path.isfile("model_checkpoints/prev_grads.pt"):
        prev_grads = torch.load("model_checkpoints/prev_grads.pt", map_location=DEVICE)
    else:
        for param in net.parameters():
            if not isinstance(prev_grads, torch.Tensor):
                prev_grads = torch.zeros_like(param.view(-1))
                prev_grads.to(DEVICE)
            else:
                prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
                prev_grads.to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    alpha = 0.01
    for _ in tqdm(range(epochs)):
        inputs,labels = next(iter(trainloader))
        inputs, labels = inputs.float().to(DEVICE), labels.long().to(DEVICE)
        output = net(inputs)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels
        
        #Dynamic Regularisation
        lin_penalty = 0.0
        curr_params = None
        for param in net.parameters():
            if not isinstance(curr_params, torch.Tensor):
                curr_params = param.view(-1)
            else:
                curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

        lin_penalty = torch.sum(curr_params * prev_grads)
        loss -= lin_penalty
        
        quad_penalty = 0.0
        for y, z in zip(net.parameters(), x.parameters()):
            quad_penalty += torch.nn.functional.mse_loss(y.data, z.data, reduction='sum')

        loss += (alpha/2) * quad_penalty

        gradients = torch.autograd.grad(loss,net.parameters())
        
        for param, grad in zip(net.parameters(),gradients):
            param.data -= lr * grad.data

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break     
        
    #Calculate the difference between updated model (y) and the received model (x)
    delta = None
    for y, z in zip(net.parameters(), x.parameters()):
        if not isinstance(delta, torch.Tensor):
            delta = torch.sub(y.data.view(-1), z.data.view(-1))
        else:
            delta = torch.cat((delta, torch.sub(y.data.view(-1), z.data.view(-1))),dim=0)

    #Update prev_grads using delta which is scaled by alpha
    prev_grads = torch.sub(prev_grads, delta, alpha = alpha)
    torch.save(prev_grads, "model_checkpoints/prev_grads.pt")
    return net
              
def train_mimelite(net, state, trainloader, epochs, deadline=None):
    #In the case of MimeLite, control_variate is nothing but a state like in case of momentum method
    x = deepcopy(net)
    
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9
    net.train()

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(net(images), labels)
            
            #Compute (full-batch) gradient of loss with respect to net's parameters 
            grads = torch.autograd.grad(loss,net.parameters())
            #Update net's parameters using gradients
            with torch.no_grad():
                for param,grad,s in zip(net.parameters(), grads, state):
                    param.data = param.data - lr * ((1-momentum) * grad.data + momentum * s.data)

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break               
    
    #Compute gradient wrt the received model (x) using the wholde dataset
    data = DataLoader(trainloader.dataset, batch_size = len(trainloader) * trainloader.batch_size, shuffle = True)  
    for images, labels in data:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = x(images)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels            
        gradient_x = torch.autograd.grad(loss,x.parameters())
    
    return net, gradient_x     

def train_mime(net, state, control_variate, trainloader, epochs, deadline=None):
    #In the case of MimeLite, control_variate is nothing but a state like in case of momentum method
    x = deepcopy(net)
    
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9
    net.train()
    x.train()

    for epoch in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(net(images), labels)
            
            #Compute (full-batch) gradient of loss with respect to net's parameters 
            grads_y = torch.autograd.grad(loss,net.parameters())

            if (epoch == 0):##Here i think epoch==0 should not be there
                output = x(images)
                loss = criterion(output, labels)
                grads_x = torch.autograd.grad(loss,x.parameters())

            #Update net's parameters using gradients
            with torch.no_grad():
                for g_y, g_x, c in zip(grads_y, grads_x, control_variate):
                    g_y.data -= g_x.data + c.to(DEVICE) 

                for param,grad,s in zip(net.parameters(), grads_y, state):
                    param.data = param.data - lr * ((1-momentum) * grad.data + momentum * s.data)

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break               
    
    #Compute gradient wrt the received model (x) using the wholde dataset
    data = DataLoader(trainloader.dataset, batch_size = len(trainloader) * trainloader.batch_size, shuffle = True)  
    for images, labels in data:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = x(images)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels            
        gradient_x = torch.autograd.grad(loss,x.parameters())
    
    return net, gradient_x            
    
def train_scaffold(net, server_c, trainloader, epochs, deadline=None):
    x = deepcopy(net)
    client_c = deepcopy(server_c)
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(net(images), labels)
            
            #Compute (full-batch) gradient of loss with respect to net's parameters 
            grads = torch.autograd.grad(loss,net.parameters())
                        
            #Update y's parameters using gradients, client_c and server_c [Algorithm line no:10]

            for param,grad,s_c,c_c in zip(net.parameters(),grads,server_c,client_c):
                s_c, c_c = s_c.to(DEVICE), c_c.to(DEVICE)
                param.data = param.data - lr * (grad.data + (s_c.data - c_c.data))
                
        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break               
            
    delta_c = [torch.zeros_like(param) for param in net.parameters()]
    new_client_c = deepcopy(delta_c)

    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data      

    a = (ceil(len(trainloader.dataset) / trainloader.batch_size) * epochs * lr)
    for n_c, c_l, c_g, diff in zip(new_client_c, client_c, server_c, net.parameters()):
        n_c.data += c_l.data - c_g.data - diff.data / a
                    
    #Calculate delta_c which equals to new_client_c-client_c
    for d_c, n_c_l, c_l in zip(delta_c, new_client_c, client_c):
        d_c.data.add_(n_c_l.data - c_l.data)

    return net, delta_c


def test_model(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader) :
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy



# trainloader, testloader, num_examples = load_data()
# model = Net().to(DEVICE)
# torch.save(model.state_dict(), "test_state_dict.pth")
# model.load_state_dict(torch.load('trained_model.pt'))
# train_model(model, trainloader, 3)
# print(test_model(model, testloader))
# # torch.save(model.state_dict(), 'trained_model.pt')
