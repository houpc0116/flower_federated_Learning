from collections import OrderedDict
import warnings
import argparse 
import wandb 
import flwr as fl
import os  
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np 
from torch.nn import GroupNorm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet18
from model import Net, mnist_Net
import ddu_dirty_mnist
from tqdm import tqdm
import torchvision
#from dataset_utils import get_cifar_10, get_mnist, do_fl_partitioning, get_dataloader

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs, args=None):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()  ##交叉熵损失函数
    if (args.strategy == 'fedavg') | (args.strategy == 'fedadagrad') :
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    elif (args.strategy == 'fedadam') | (args.strategy == 'fedyogi') : 
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters())
    net.train()
    
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            wandb.log({"train_loss": loss})
            loss.backward()
            optimizer.step()

## Load Data
## 下一階段 partition
def load_data(args=None):
    """Load CIFAR-10 (training and test set)."""
#    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if args.datasets == 'cifar' : 
        trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
        testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    elif args.datasets == 'cifar100' : 
        trainset = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
        testset = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    elif args.datasets == 'mnist' :
        trainset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        testset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    

    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

## 2022/10/28 新增 ##
"""
Data loading
Let's now load the CIFAR-10 training and test set, 
partition them into ten smaller datasets (each split into training and validation set), 
and wrap everything in their own DataLoader. 
We introduce a new parameter num_clients which allows us to call load_datasets with different numbers of clients.
"""
def load_datasets(num_clients, args=None):
    # Download and transform CIFAR-10 (train and test)
    cifar_transform = transforms.Compose(
        [transforms.transforms.RandomCrop(24),
         transforms.transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), 
         transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    mnist_transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if args.datasets == 'cifar' : 
        trainset = CIFAR10("./dataset", train=True, download=True, transform=cifar_transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=cifar_transform)
    elif args.datasets == 'cifar100' : 
        trainset = CIFAR100("./dataset", train=True, download=True, transform=cifar_transform)
        testset = CIFAR100("./dataset", train=False, download=True, transform=cifar_transform)
    elif args.datasets == 'mnist' :
        trainset = MNIST("./dataset", train=True, download=True, transform=mnist_transform)
        testset = MNIST("./dataset", train=False, download=True, transform=mnist_transform)
    elif args.datasets == 'dirty' :
        trainset = ddu_dirty_mnist.DirtyMNIST("./dirty_mnist", 
                                                        train=True, 
                                                        download=True, 
                                                        )
        trainset = ddu_dirty_mnist.DirtyMNIST("./dirty_mnist", 
                                                        false=True, 
                                                        download=True, 
                                                        )
    #trainset = MNIST("./dataset", train=True, download=True, transform=mnist_transform)
    #testset = MNIST("./dataset", train=False, download=True, transform=mnist_transform)

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    """Create model, load data, define Flower client, start Flower client."""
    wandb.init(project="nilm", entity="josie_hou", group='mnist_fedavg', job_type='eval2')

    parser = argparse.ArgumentParser(description="Flower-Client")
    #parser.add_argument("--partition", type=int, choices=range(0, 100), required=True)
    parser.add_argument("--partition", type=int, default=5, required=True)## partition [5 6 7 8]
    parser.add_argument("--datasets", type=str, default='cifar100')
    parser.add_argument("--model", type=str, default='cnn')
    parser.add_argument("--lr", type=float, default=0.01)
    #parser.add_argument("--num_gpu", type=int, default=0)
    parser.add_argument("--client", type=int, default=5)
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='fedavg')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--num_client_cpus", type=int, default=2)  ##CPU核心
    parser.add_argument("--num_rounds", type=int, default=5)
    args = parser.parse_args()

    # GPU/CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.model == 'cnn' : 
        if (args.datasets == 'mnist') | (args.datasets == 'dirty') : 
            net = mnist_Net()
        elif args.datasets == 'cifar10' : 
            net = Net() 

    elif args.model == 'resnet' : 
        net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100)

    wandb.config.update(args)
    wandb.watch(net)
    
    trainloader, testloader = load_data(args=args)
    #trainloaders, valloaders, testloader = load_datasets(5, args=args)##num_clients, args
    #print(trainloaders)
    # Flower client
    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, args):
            if args.model == 'cnn' :
                if (args.datasets == 'mnist') | (args.datasets=='dirty') : 
                    net = mnist_Net()
                    #net = mnist_Net().cuda()
                elif args.datasets == 'cifar10' : 
                    net = Net()
                    #net = Net().cuda()
            elif args.model == 'resnet18': 
                 self.net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100)
                 #self.net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100).cuda()

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=args.local_ep, args=args)
            return self.get_parameters(), len(trainloader), {}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=FlowerClient(args=args))


if __name__ == "__main__":
    main()