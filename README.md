# fl_framework_initial
_ is a highly dynamic and customizable framework that can accommodate many use cases with flexibility by implementing several functionalities over different federated learning algorithms, and essentially creating a plug-and-play architecture to accommodate different use cases.

## The framework will be composed of 4 phases, each phase building upon the last:

* **Phase 1: FedAvg**. The framework will be able to efficiently coordinate a server and some clients for a specific number of communication rounds, with granularity in selection of clients included.
* **Phase 2: Verification**. Before aggregating, the server will perform a special verification round to determine which models to accommodate during aggregation.
* **Phase 3: Timeout**. Instead of waiting indefinitely for a client to finish training, the server will be able to issue a timeout, upon the completion of which, even if it hasn’t completed all epochs, the client will stop training and return the results.
* **Phase 4: Intermediate client connections**. New clients will be able to join the server anytime and may even be included in a round that is already live.

## Verification module 

* After the server receives the trained weights, it aggregates all of them to form the new model. However, the selection of models for aggregation can be modified.
* Before aggregation, the server passes the models to a Verification module, which then uses a predefined procedure to generate scores for models, and then returns only those models that have performed above a defined threshold
* The Verification module can be easily customized.
<p align="center">
<img src = "./media/verification module.png" width=650>
</p>

## Training timeouts module

* Often in real world scenarios, clients cannot keep training indefinitely. Therefore, a timeout functionality has been implemented.
* The server can specify a timeout parameter as a Train order configuration. The client will then train till the timeout occurs, and then return the results.
<p align="center">
<img src = "./media/timeout module.png" width=650>
</p>





