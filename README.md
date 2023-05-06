# Federated Learning Framework

[![License](.media/license.svg)](https://opensource.org/licenses/Apache-2.0)

It is a highly dynamic and customizable framework that can accommodate many use cases with flexibility by implementing several functionalities over different federated learning algorithms, and essentially creating a plug-and-play architecture to accommodate different use cases.

## Supported Devices

Feder has been extensively tested on and works with the following devices:

* Intel CPUs
* Nvidia GPUs
* Nvidia Jetson
* Raspberry Pi
* Intel NUC

With Feder, it is possible to operate the server and clients on **separate devices** or on a **single device** through various means, such as utilizing different terminals or implementing multiprocessing.

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

* **Module 1: Verification module** [docs](https://feder.readthedocs.io/en/latest/overview.html#verification-module)
* **Module 2: Timeout module** [docs](https://feder.readthedocs.io/en/latest/overview.html#timeout-module)
* **Module 3: Intermediate client connections module** [docs](https://feder.readthedocs.io/en/latest/overview.html#intermediate-client-connections-module)
* **Module 4: Carbon emission tracking module** [docs](https://feder.readthedocs.io/en/latest/overview.html#carbon-emissions-tracking-module)

## Running tests

Various unit tests are available in the `test` directory. To run any tests, run the following command from the root directory:

```
cd test
python test_algorithms.py
```

## Federated Learning Algorithms

Following federated learning algorithms are implemented in this framework:

| Method              | Paper                                                        | Publication                                     |
| ------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| FedAvg              | [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) | AISTATS'2017 |                                                      
| FedDyn              | [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) | ICLR' 2021   |          
| Scaffold           | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning]() | ICML'2020    |
| Personalized FedAvg | [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/pdf/1909.12488.pdf) |    Pre-print      |                                                      
| FedAdagrad          | [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf) | ICML'2020    |                                                       
| FedAdam       | [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf) | ICML'2020    |                                                      
| FedYogi    | [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf) | ICML'2020    |                                                      
| Mime       | [Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning](https://arxiv.org/pdf/2008.03606.pdf) | ICML'2020    |                                                      
| Mimelite       | [Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning](https://arxiv.org/pdf/2008.03606.pdf) | ICML'2020    |                                                      

## Datasets & Data Partition

Sophisticated in the real world, FL needs to handle various kind of data distribution scenarios, including iid and non-iid scenarios. Though there already exists some datasets and partition schemes for published data benchmark, it still can be very messy and hard for researchers to partition datasets according to their specific research problems, and maintain partition results during simulation.

### Data Partition

We provide multiple data partition schemes used in recent FL papers[[1]](#1)[[2]](#2)[[3]](#3). Here we show the data partition visualization of several common used datasets as the examples.

#### Balanced IID partition

Each client has same number of samples, and same distribution for all class samples. 

#### Non-IID partition 2

#### Non-IID partition 3

#### Non-IID partition 4

#### Non-IID partition 5

### Datasets Supported

| Dataset                | Training samples         | Test samples       | Classes 
| ---------------------- | ------------------------ | ------------------ | ------------------ |
| MNIST                  | 60,000                   | 10,000             | 10                 |
| CIFAR-10               | 50,000                   | 10,000             | 10                 |
| CIFAR-100              | 50,000                   | 10,000             | 100                |
| FashionMnist           | 60,000                   | 10,000             | 10                 |

### Custom Dataset Support

We also provide a simple way to add your own dataset to the framework. Look into [docs](https://feder.readthedocs.io/en/latest/tutorials/dataset.html#adding-support-for-new-datasets) for more details.

## Models Supported

**Feder** has support for the following Deep Learning models, which are loaded from `torchvision.models`:

* LeNet
* ResNet18
* ResNet50
* VGG16
* AlexNet

### Custom Model Support

We also provide a simple way to add your own models to the framework. Look into [docs](https://feder.readthedocs.io/en/latest/tutorials/dataset.html#adding-support-for-new-datasets) for more details.

## Carbon emission tracking

In Feder CodeCarbon package is used to estimate the carbon emissions generated by clients during training. CodeCarbon is a Python package that provides an estimation of the carbon emissions associated with software code.

### Nvidia gpu

---

### raspberry pi

---

## Performance Evaluation under different Non-IID setting

### Accuracy of various FL algorithms available in the framework with a few dataset

<p align="center">
<table style="margin: auto">
    <thead>
        <tr>
            <th>Dataset</th>
            <th colspan=3>MNIST</th>
            <th colspan=3>CIFAR-10</th>
            <th colspan=3>CIFAR-100</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td  style="text-align: center" >Algorithm </td>
            <td  style="text-align: center">K=1</td>
            <td  style="text-align: center">K=2</td>
            <td  style="text-align: center">K=3</td>
            <td  style="text-align: center">K=1</td>
            <td  style="text-align: center">K=2</td>
            <td  style="text-align: center">K=3</td>
            <td  style="text-align: center">K=1</td>
            <td  style="text-align: center">K=2</td>
            <td  style="text-align: center">K=3</td>
        </tr>
        <tr>
            <td>FedAvg </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>FedAvgM </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
         <tr>
            <td>FedAdam </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
         <tr>
            <td>FedAdagrad </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
         <tr>
            <td>FedYogi </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
 <tr>
            <td>Mime </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Mimelite </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
         <tr>
            <td>FedDyn </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
         <tr>
            <td>Scaffold </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
         <tr>
            <td>Personalized-FedAvg </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
             <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>
</p>


### Visualizing the accuracy os some algorithms against different Non-IID distributions

<p float="left">
  <img src="media/Al_0.png" width="210" />
  <img src="media/Al_1.png" width="210" /> 
  <img src="media/Al_3.png" width="210" />
  <img src="media/Al_4.png" width="210" />
</p>

### Plotting accuracy of different algorithms for MNIST with different Non-IID distribution

<p float='left'>
  <img src="media/niid_1.png" width="220" />
  <img src="media/niid_2.png" width="220" />
  <img src="media/niid_3.png" width="220" />
  <img src="media/niid_4.png" width="220" />
</p>

<br/>

### Comparing accuracy of different algorithm with different Non-IID distributions

<div align="center">
  <img width="60%" alt="" src="media/Accuracy.png" >
</div>

<br/>

## References

<a id="1">[1]</a> Schmidt, V., Goyal, K., Joshi, A., Feld, B., Conell, L., Laskaris, N., Blank, D., Wilson, J., Friedler, S., & Luccioni, S. (2021). CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing. https://doi.org/10.5281/zenodo.4658424

<a id="2">[2]</a>

## Contact

Project Investigator: [Prof. ](https://scholar.google.com/citations?user=gF0H9nEAAAAJ&hl=en) (xuzenglin@hit.edu.cn).

For technical issues related to __Feder__ development, please contact our development team through Github issues or email:

- [Name Sirname](https://scholar.google.com/citations): _____@gmail.com
