### 版本 ver 1.0  日期:2022.10.29

修改成CPU執行


# Federated_Learning
Implementation of federated learning using flower framework

##  Quickstart 

### Weights & Biases(Visualization tool)

- Before starting, you should login wandb using your personal API key. 
- Weights & Biases : https://wandb.ai/site

```shell
!pip install wandb
wandb login PERSONAL_API_KEY
```

### Install flower library

```shell
!pip install flwr
```

### Cloning a repository

```shell
git clone https://github.com/ssuncheol/federated_Learning.git
```

### Flwr folder structure as follows 


```
client
 └─── app.py           
 └─── client.py
 └─── numpy_client.py 
 
server
 └─── app.py
 └─── client_manager.py
 └─── client_proxy.py
 └─── criterion.py
 └─── history.py
 └─── server.py 
 
 └─── strategy
          └───── __init__.py
          └───── aggregate.py
          └───── default.py 
          └───── fedavg.py
          └───── fedopt.py 
          └───── ...
        
```

### Experiment settings 

```
Add dataset(mnist, cifar10), model(cifar-cnn,resnet18 with groupnorm)
```

```
Add client-side setting, server-side setting (Strategy : fedavg, fedadam, fedadagrad, fedyogi) 
```


### How to train 

- multi-gpu setting
```
sh run.sh 
```

- single-gpu setting 
```
sh run2.sh 
```


