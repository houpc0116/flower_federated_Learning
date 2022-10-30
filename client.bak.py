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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet18
from model import Net, mnist_Net
import ddu_dirty_mnist 
from dataset_utils import get_cifar_10, get_mnist, do_fl_partitioning, get_dataloader

warnings.filterwarnings("ignore", category=UserWarning)

wandb.init(project="nilm", entity="josie_hou", group='mnist_fedavg', job_type='eval2')

def train(net, trainloader, epochs, args=None):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
#    criterion = torch.nn.CrossEntropyLoss().cuda()
    if (args.strategy == 'fedavg') | (args.strategy == 'fedadagrad') :
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    elif (args.strategy == 'fedadam') | (args.strategy == 'fedyogi') : 
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)    
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images, labels
#            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            wandb.log({"train_loss": loss})
            loss.backward()
            optimizer.step()
    del net 


def load_data(client, partition, args=None):
    """Load CIFAR-10 (training and test set)."""
    cifar_transform = transforms.Compose(
        [transforms.transforms.RandomCrop(24),
         transforms.transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), 
         transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.transforms.Normalize((0.5, ), (0.5, ))]
    )
    if args.datasets == 'cifar' : 
        trainset = CIFAR10("./dataset", train=True, download=True, transform=cifar_transform)

    if args.datasets == 'cifar100' : 
        trainset = CIFAR100("./dataset", train=True, download=True, transform=cifar_transform)

    elif args.datasets == 'mnist' :
        trainset = MNIST("./dataset", train=True, download=True, transform=mnist_transform)
        
    #num_examples = {'trainset' : len(trainset)}

    if args.datasets == 'dirty' : 
        dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST("./dirty_mnist", 
                                                        train=True, 
                                                        download=True, 
                                                        )
        clean, ambig = dirty_mnist_train.datasets[0], dirty_mnist_train.datasets[1]
    
        num_examples = {"trainset" : len(dirty_mnist_train)}                                        
    
    # Data Partitioning 
    temp = tuple([len(clean)//client for i in range(client)])
    dirty_temp = tuple([len(ambig)//client for i in range(client)])
    
    Worker_data = torch.utils.data.random_split(clean, temp)
    Dirty_worker_data = torch.utils.data.random_split(ambig, dirty_temp)

    if partition in [5,6,7,8,9] :
        trainloader = torch.utils.data.DataLoader(Worker_data[partition], batch_size=64,
                                              shuffle=True, num_workers=2)

         
    else : 
        data_concat = torch.from_numpy(np.vstack((Worker_data[partition][:3000][0], Dirty_worker_data[partition][:3000][0])))
        label_concat = torch.from_numpy(np.hstack((Worker_data[partition][:3000][1], Dirty_worker_data[partition][:3000][1])))
        #dirty_dataset = (data_concat, label_concat)
        dirty_dataset = torch.utils.data.TensorDataset(data_concat, label_concat)

        
        #dirty_dataset = torch.utils.data.dataset.ConcatDataset([Worker_data[partition][:3000], Dirty_worker_data[partition][:3000]])
        trainloader = torch.utils.data.DataLoader(dirty_dataset, batch_size=64,
                                               shuffle=True, num_workers=2)

    

    #import pdb; pdb.set_trace()

    print(f'{partition}th client train data is {len(trainloader)*64}')
    
    return trainloader, num_examples



# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    """Create model, load data, define Flower client, start Flower client."""
    
    parser = argparse.ArgumentParser(description="Flower-Client")
#    parser.add_argument("--partition", type=int, choices=range(0, 100), required=True)
    parser.add_argument("--partition", type=int, default=100, required=True)
    parser.add_argument("--datasets", type=str, default='cifar100')
    parser.add_argument("--model", type=str, default='cnn')
    parser.add_argument("--lr", type=float, default=0.01)
#    parser.add_argument("--num_gpu", type=int, default=0)
    parser.add_argument("--client", type=int, default=10)
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='fedavg')
#    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--num_client_cpus", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    args = parser.parse_args()

#    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   
    pool_size = 100  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs
    
    # Download CIFAR-10 / MNIST dataset
    if args.datasets == 'cifar10':
       train_path, testset = get_cifar_10()
    elif args.datasets == 'mnist':
       train_path, testset = get_mnist()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )

    if args.model == 'cnn' : 
        if (args.datasets == 'mnist') | (args.datasets == 'dirty'): 
            net = mnist_Net()
#           net = mnist_Net().cuda()
        elif args.datasets == 'cifar10': 
            net = Net()
#            net = Net().cuda() 
    elif args.model == 'resnet': 
        net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100)

#        net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100).cuda()
    
    wandb.config.update(args)
    wandb.watch(net)

    # Load data (CIFAR-10) client, partition
#    trainloader, num_examples = load_data(args.client*num_client_cpus, args.partition, args=args)

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
#                self.net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100).cuda()

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, train_path, epochs=args.local_ep, args=args)
            return self.get_parameters(), len(train_path), {}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=FlowerClient(args=args))


if __name__ == "__main__":
    main()