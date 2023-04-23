
import argparse
import torch

from .src.server import server_start
from .src.server_lib import save_intial_model

#the parameters that can be passed while starting the server 
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default = "fedavg", help="Aggregation algorithm")
parser.add_argument("--clients", type=int, default = 1, help="Number of clients to select")
parser.add_argument("--fraction", type = float, default = None, help = "Fraction of clients to select out of the number provided or those available. Float between 0 to 1 inclusive")
parser.add_argument("--rounds", type=int, default = 1, help = "Total number of communication rounds to perform")
parser.add_argument("--model_path", default = "server/src/initial_model.pt", help = "The path of the initial server model's state dict")
parser.add_argument("--epochs", type = int, default = 1, help="Number of epochs each client should perform in each round")
parser.add_argument("--accept_conn", type = int, default = 1, help = "If set to 1, connections will be accpeted even after FL has begun, else if set to 0, they will be rejected.")
parser.add_argument("--verify", type = int, default = 0, help="If the verification module should be run before each round. Specify 1 for True or 0 for False")
parser.add_argument("--threshold", type = float, default = 0, help = "Minimum score clients must have in a verification round, if verification is enabled. Between 0 and 1 inclusive")
parser.add_argument("--timeout", type = int, default=None, help="The time limit each client has when training during each round. Specified in seconds")
parser.add_argument("--resize_size", type = int, default = 32, help="The resize dimension")
parser.add_argument("--batch_size", type = int, default = 32, help="The batch size")
parser.add_argument("--net", type = str, default = "LeNet", help="The network architecture")    
parser.add_argument("--dataset", type = str, default= "MNIST", help="The datsset ")
parser.add_argument("--niid", type = int,default=2, help="value should be [0, 5]")
parser.add_argument("--carbon", type = int,default=0, help="set to 1 if carbon emission need to be track at client side")
args = parser.parse_args()

configurations = {
    "algorithm": args.algorithm,
    "num_of_clients": args.clients,
    "fraction_of_clients": args.fraction,
    "num_of_rounds": args.rounds,
    "initial_model_path": args.model_path,
    "epochs": args.epochs,
    "accept_conn_after_FL_begin": args.accept_conn,
    "verify": args.verify,
    "verification_threshold": args.threshold,
    "timeout": args.timeout,
    "resize_size": args.resize_size,
    "batch_size": args.batch_size,
    "net": args.net,
    "dataset": 'MNIST',
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "niid": args.niid,
    "carbon":args.carbon,
}

#start the server with the given parameters
if __name__ == '__main__':
    save_intial_model(configurations)
    server_start(configurations) 