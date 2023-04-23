import argparse
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from server.src.server import server_start 

parser = argparse.ArgumentParser()
parser.add_argument('--infile', nargs=1,
                    help="JSON file to be processed",
                    type=argparse.FileType('r'))
arguments = parser.parse_args()

# Loading a JSON object returns a dict.
d = json.load(arguments.infile[0])
d = d['fedavg']
server_start(d)