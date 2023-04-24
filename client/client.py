
import json
from queue import Queue
import torch
from io import BytesIO
import time

import grpc
import ClientConnection_pb2_grpc
from ClientConnection_pb2 import ClientMessage

from client_lib import train, evaluate, set_parameters

keep_going = True
wait_time = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

while keep_going:
    #wait for specified time before reconnecting
    time.sleep(wait_time)
    
    #create new gRPC channel to the server
    channel = grpc.insecure_channel("localhost:8214", options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ])
    stub = ClientConnection_pb2_grpc.ClientConnectionStub(channel)
    client_buffer = Queue(maxsize = 1)

    #wait for incoming messages from the server in client_buffer
    #then according to fields present in them call the appropraite function
    for server_message in stub.Connect( iter(client_buffer.get, None) ):
        if server_message.HasField("evalOrder"):
            eval_order_message = server_message.evalOrder
            eval_response_message = evaluate(eval_order_message)
            message_to_server = ClientMessage(evalResponse = eval_response_message)
            client_buffer.put(message_to_server)

        if server_message.HasField("trainOrder"):
            train_order_message = server_message.trainOrder
            train_response_message = train(train_order_message)
            message_to_server = ClientMessage(trainResponse = train_response_message)
            client_buffer.put(message_to_server)

        if server_message.HasField("setParamsOrder"):
            set_parameters_order_message = server_message.setParamsOrder
            set_parameters_response_message = set_parameters(set_parameters_order_message)
            message_to_server = ClientMessage(setParamsResponse = set_parameters_response_message)
            client_buffer.put(message_to_server)

        if server_message.HasField("disconnectOrder"):
            print("Current FL process is done ")
            disconnect_order_message = server_message.disconnectOrder
            message = disconnect_order_message.message
            print(message)
            reconnect_time = disconnect_order_message.reconnectTime
            if reconnect_time == 0:
                keep_going = False
                break
            wait_time = reconnect_time
            