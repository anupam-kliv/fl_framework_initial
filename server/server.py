from client_manager import ClientManager
from client_connection_servicer import ClientConnectionServicer

from server_evaluate import server_eval

import grpc
import ClientConnection_pb2_grpc
from concurrent import futures

import os
import json
import threading
import torch
from collections import OrderedDict
from datetime import datetime

#the business logic of the server, i.e what interactions take place with the clients
def server_runner(client_manager, configurations):
    print("\nStarted server runner")

    #get hyperparameters from the passed configurations dict
    config_dict = {"message": "eval"}
    algorithm = configurations["algorithm"]
    num_of_clients = configurations["num_of_clients"]
    fraction_of_clients = configurations["fraction_of_clients"]
    clients = client_manager.random_select(num_of_clients, fraction_of_clients) 
    communication_rounds = configurations["num_of_rounds"]
    initial_model_path = configurations["initial_model_path"]
    server_model_state_dict = torch.load(initial_model_path, map_location="cpu")
    epochs = configurations["epochs"]
    accept_conn_after_FL_begin = configurations["accept_conn_after_FL_begin"]

    #create a new directory inside FL_checkpoints and store the aggragted models in each round
    fl_timestamp = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    save_dir_path = f"FL_checkpoints/{fl_timestamp}"
    os.mkdir(save_dir_path)
    torch.save(server_model_state_dict, f"{save_dir_path}/initial_model.pt")
    #create new file inside FL_results to store training results
    with open(f"{save_dir_path}/FL_results.txt", "w") as file:
        pass
    print(f"FL has started, will run for {communication_rounds} rounds...")
    
    #Initialize the aggregation algorithm
    exec(f"from algorithms.{algorithm} import {algorithm}")
    aggregator = eval(algorithm)(configurations)
    print(f"{algorithm} is used for aggregation")
    
    #If the algorithm is either scaffold or mimelite, then we need to make use of control variate
    if (algorithm == 'scaffold' or algorithm == 'mimelite'):
        control_variate = [torch.zeros_like(server_model_state_dict[key]) for key in server_model_state_dict.keys()] #At initialization, control variate should be zero as mentioned in the paper
        print("Control Variate is used")
    else:
        control_variate = None
        print("Control Variate is not in use")
    
    #To check the control_variate
    #print(control_variate)
        
    #run FL for given rounds
    client_manager.accepting_connections = accept_conn_after_FL_begin
    for round in range(1, communication_rounds + 1):
        clients = client_manager.random_select(client_manager.num_connected_clients(), fraction_of_clients) 
        
        # for client in clients:
        #     print(client.client_id)
        #     results = client.evaluate( server_model_state_dict, config_dict )
        #     print(results)
        
        # for client in clients:
        #     client.set_parameters(server_model_state_dict)
        
        print(f"Communication round {round} is starting with {len(clients)} client(s) out of {client_manager.num_connected_clients()}.")
        config_dict = {"message": "train", "epochs": epochs, "algorithm":algorithm}
        trained_model_state_dicts = []
        updated_control_variates = []
        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            result_futures = {executor.submit(client.train, server_model_state_dict, control_variate, config_dict) for client in clients}
            for client_index, result_future in zip(range(len(clients)), futures.as_completed(result_futures)):
                trained_model_state_dict, updated_control_variate, results = result_future.result()
                trained_model_state_dicts.append(trained_model_state_dict)
                updated_control_variates.append(updated_control_variate)
                print(f"Training results (client {clients[client_index].client_id}): ", results)
        print("Recieved all trained model parameters.")
        
        selected_state_dicts = trained_model_state_dicts
        
        #aggregate model, save it, then send to some client to evaluate
        if control_variate:
            server_model_state_dict, control_variate = aggregator.aggregate(server_model_state_dict, control_variate, selected_state_dicts, updated_control_variates)
        else:
            server_model_state_dict = aggregator.aggregate(server_model_state_dict,selected_state_dicts)
        
        torch.save(server_model_state_dict, f"{save_dir_path}/round_{round}_aggregated_model.pt")

        #test on server test set
        print("Evaluating on server test set...")
        eval_result = server_eval(server_model_state_dict)
        eval_result["round"] = round
        print("Eval results: ", eval_result)
        #store the results
        with open(f"{save_dir_path}/FL_results.txt", "a") as file:
            file.write( str(eval_result) + "\n" )
        
        #To check the control_variate
        #print(control_variate)
        
    # #sync all connected clients with current global model and order them to disconnect
    for client in client_manager.random_select():
        client.set_parameters(server_model_state_dict)
        client.disconnect()
    torch.save(server_model_state_dict, initial_model_path)
    print("Server runner stopped.")

#starts the gRPC server and then runs server_runner concurrently
def server_start(configurations):
    client_manager = ClientManager()
    client_connection_servicer = ClientConnectionServicer(client_manager)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ClientConnection_pb2_grpc.add_ClientConnectionServicer_to_server( client_connection_servicer, server )
    server.add_insecure_port('localhost:8213')
    server.start()

    server_runner_thread = threading.Thread(target = server_runner, args = (client_manager, configurations, ))
    server_runner_thread.start()
    server_runner_thread.join()
    

    server.stop(None)
