# Federated Learning Framework

[![License](.media/license.svg)](https://opensource.org/licenses/Apache-2.0)

It is a highly dynamic and customizable framework that can accommodate many use cases with flexibility by implementing several functionalities over different federated learning algorithms, and essentially creating a plug-and-play architecture to accommodate different use cases.

## Installation

- Install the latest version from source code:
```
$ git clone git@github.com:feder.git
$ cd Feder
$ pip install -r requirements.txt
```

- Install the stable version (old version) via pip:
```
# assign the version feder==1.1.2
$ pip install feder
```

## Starting server

```
python -m server.start_server \
 --algorithm fedavg \
 --clients 2 \
 --rounds 10 \
 --epochs 10 \
 --batch_size 10 \
 --lr 0.01 \
 --dataset mnist \
```

## Starting client

```
cd client
python client.py
```

## Architecture
Files architecture of Feder. These contents may be helpful for users to understand our repo.

```
├── client
│   ├── src
|   |   ├── client_lib
|   |   ├── client
|   |   ├── ClientConnection_pb2_grpc
|   |   ├── ClientConnection_pb2
|   |   ├── data_utils
|   |   ├── distribution
|   |   ├── get_data
|   |   ├── net_lib
|   |   ├── net
│   └── start_client
├── server
│   ├── src
|   |   ├── algorithms
|   |   ├── server_evaluate
|   |   ├── client_connection_servicer
|   |   ├── client_manager
|   |   ├── client_wrapper
|   |   ├── ClientConnection_pb2_grpc
|   |   ├── ClientConnection_pb2
|   |   ├── server_lib
|   |   ├── server
|   |   ├── verification
│   └── start_server
├── test
|   ├── misc
│   ├── test_algorithms
|   ├── test_datasets
│   ├── test_models
│   ├── test_modules
│   ├── test_results
│   └── test_scalability
└── tutorials
    ├── Code_Carbon_Tutorial.ipynb
    └── ...
```

## The framework will be composed of 4 modules, each module building upon the last:

* **Module 1: Verification module**[docs](https://feder.readthedocs.io/en/latest/overview.html#verification-module)
* **Module 2: Timeout module**[docs](https://feder.readthedocs.io/en/latest/overview.html#timeout-module)
* **Module 3: Intermediate client connections module**[docs](https://feder.readthedocs.io/en/latest/overview.html#intermediate-client-connections-module)
* **Module 4: Carbon emission tracking module**[docs](https://feder.readthedocs.io/en/latest/overview.html#carbon-emissions-tracking-module)

## Running tests

Various unit tests are available in the `test` directory. To run any tests, run the following command from the root directory:

```
cd test
python test_algorithms.py
```