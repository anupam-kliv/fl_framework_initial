
import torch
from io import BytesIO
import json
import time
import os 
from datetime import datetime

from net import get_net
from net_lib import test_model, load_data, flush_memory, DEVICE
from net_lib import train_model, train_fedavg, train_scaffold, train_mimelite, train_mime, train_feddyn

from ClientConnection_pb2 import  EvalResponse, TrainResponse

# Load data
# trainloader, testloader, num_examples = load_data()
# print("data was loaded")

# # Load model
# model = Net().to(DEVICE)

#create a new directory inside FL_checkpoints and store the aggragted models in each round
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fl_timestamp = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
save_dir_path = f"model_checkpoints/{fl_timestamp}"
os.mkdir(save_dir_path)
# torch.save(model.state_dict(), f"{save_dir_path}/initial_model.pt")

#the different functions that can be performed by the client

def evaluate(eval_order_message):
    model_parameters_bytes = eval_order_message.modelParameters
    model_parameters = torch.load( BytesIO(model_parameters_bytes), map_location="cpu" )

    config_dict_bytes = eval_order_message.configDict
    config_dict = json.loads( config_dict_bytes.decode("utf-8") )
    print(config_dict["message"])
    
    state_dict = model_parameters
    model = get_net(config= config_dict)
    model.load_state_dict(state_dict)
    _, testloader, _ = load_data(config_dict)
    eval_loss, eval_accuracy = test_model(model, testloader)

    response_dict = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    response_dict_bytes = json.dumps(response_dict).encode("utf-8")
    eval_response_message = EvalResponse(responseDict = response_dict_bytes)
    return eval_response_message


def train(train_order_message):
    data_bytes = train_order_message.modelParameters
    data = torch.load( BytesIO(data_bytes), map_location="cpu" )
    model_parameters, control_variate, control_variate2 = data['model_parameters'], data['control_variate'], data['control_variate2']
    
    config_dict_bytes = train_order_message.configDict
    config_dict = json.loads( config_dict_bytes.decode("utf-8") )
    print(config_dict["message"])

    model = get_net(config= config_dict)
    model.load_state_dict(model_parameters)
    model = model.to(device)
    epochs = config_dict["epochs"]
    if config_dict["timeout"]:
        deadline = time.time() + config_dict["timeout"]
    else:
        deadline = None
    trainloader, testloader, _ = load_data(config_dict)
    if (config_dict['algorithm'] == 'mimelite'):
        model, control_variate = train_mimelite(model, control_variate, trainloader, epochs, deadline)
    elif (config_dict['algorithm'] == 'scaffold'):
        model, control_variate = train_scaffold(model, control_variate, trainloader, epochs, deadline)
    elif (config_dict['algorithm'] == 'mime'):
        model, control_variate = train_mime(model, control_variate, control_variate2, trainloader, epochs, deadline)
    elif (config_dict['algorithm'] == 'fedavg'):
        model = train_fedavg(model, trainloader, epochs, deadline)
    elif (config_dict['algorithm'] == 'feddyn'):
        model = train_feddyn(model, trainloader, epochs, deadline)
    else:
        model = train_model(model, trainloader, epochs, deadline)

    myJSON = json.dumps(config_dict)
    with open("config.json", "w") as jsonfile:
        jsonfile.write(myJSON)
        print("config file saved!")
    
    trained_model_parameters = model.state_dict()
    #Create a dictionary where model_parameters and control_variate are stored which needs to be sent to the server
    data_to_send = {}
    data_to_send['model_parameters'] = trained_model_parameters
    data_to_send['control_variate'] = control_variate #If there is no control_variate, this will become None
    buffer = BytesIO()
    torch.save(data_to_send, buffer)
    buffer.seek(0)
    data_to_send_bytes = buffer.read()   

    print("train eval")
    train_loss, train_accuracy = test_model(model, testloader)
    response_dict = {"train_loss": train_loss, "train_accuracy": train_accuracy}
    response_dict_bytes = json.dumps(response_dict).encode("utf-8")

    train_response_message = TrainResponse(
        modelParameters = data_to_send_bytes, 
        responseDict = response_dict_bytes)

    save_model_state(model)
    return train_response_message

#replace current model with the model provided
def set_parameters(set_parameters_order_message):
    model_parameters_bytes = set_parameters_order_message.modelParameters
    model_parameters = torch.load( BytesIO(model_parameters_bytes), map_location="cpu" )
    with open("config.json", "r") as jsonfile:
        config_dict = json.load(jsonfile)
    model = get_net(config= config_dict).to(device)
    model.load_state_dict(model_parameters)
    save_model_state(model)

#save the current model to model_checkpoints
def save_model_state(model):
    file_num = len(os.listdir(f"{save_dir_path}"))
    filepath = f"{save_dir_path}/model_{file_num}.pt"
    state_dict = model.state_dict()
    torch.save(state_dict, filepath)