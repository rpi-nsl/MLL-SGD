import pandas as pd
from mnist import MNIST
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from emnist import extract_training_samples
from emnist import extract_test_samples
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

learning_rate = 0.01

import subprocess as sp
import os

def get_gpu_memory():
    """Helper function for seeing GPU memory used"""
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values

# From: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?fbclid=IwAR1GVudrnJd_2JyESsKWqUANKQknUDs5KWiXuZPHuTSQSLX-jLUjjAg1jkY
class CIFARNet(nn.Module):
    """
    5-layer CNN for CIFAR data. Unused in experiments, can be used input
    set to model type 2.
    """
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class cNet(nn.Module):
    """
    Convolutional Neural Network model class, used with input set to model type 1.
    Should be paired with EMNIST data, set to data type 1.
    """
    def __init__(self):
        super(cNet, self).__init__()
        self.N0 = 7*7*20
        self.conv1 = nn.Conv2d(1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, 5, padding=2)
        self.fc1 = nn.Linear(self.N0, 100)
        self.fc2 = nn.Linear(100, 62)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, self.N0)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def sigmoid(z):
    """Sigmoid activation function, used for MNIST data."""
    return 1 / (1 + np.exp(-z))

def cost_reg(theta, x, y):
    """
    Calculate loss of logsitic regression model
    """
    # one vs. rest binary classification
    conditions = False
    for c in classes[0:int(len(classes)/2)]:
        conditions = conditions | (y == c)
    y = np.where(conditions, 1, 0)

    h = sigmoid(x @ theta)
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h)
    )
    grad = 1 / m * ((y - h) @ x)
    return cost, grad

def cost_nn(model, x, y):
    """
    Calculate loss of a Neural Network
    """
    global BATCH_SIZE

    # Transfer data to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  
    num_batches = int(x.shape[0]/BATCH_SIZE)
    total_loss = 0

    # Iterate over all batches
    for b in range(num_batches):
        batch_indices = list(range(b*BATCH_SIZE, (b+1)*BATCH_SIZE))
        batch_X = x[batch_indices]
        batch_Y = y[batch_indices]
        
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()

        # Predict and calculate loss on batch
        pred_Y = model(batch_X)
        loss = criterion(pred_Y, batch_Y)
        total_loss += loss.item()

    # Move data back to CPU
    device = torch.device("cpu")
    model.to(device)
    return total_loss/num_batches, None

def fit_nn(x, y, net, max_iter=500, alpha=0.2, batch_size=-1):
    global learning_rate
    """
    Train a Neural Network
    """
    criterion = nn.CrossEntropyLoss() 
    costs = np.zeros(max_iter)
    grad = np.zeros((max_iter, x.shape[1]))

    # Transfer data to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    num_batches = 1
    if batch_size > 0:
        num_batches = int(x.shape[0]/batch_size)

    # Train for max_iter epochs
    for epoch in range(max_iter):
        indices = torch.randperm(x.shape[0])
        indices = np.array(indices)
        # Iterate over each mini-batch
        for b in range(num_batches):
            batch_indices = indices[b*batch_size: (b+1)*batch_size]
            batch_X = x[batch_indices]
            batch_Y = y[batch_indices]
            
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()

            pred_Y = net(batch_X)
            loss = criterion(pred_Y, batch_Y)
            
            # Back-propagate
            net.zero_grad()
            loss.backward()

            # Update the parameters.
            for param in net.parameters():
                param.data -= learning_rate * param.grad.data

    # Move data back to CPU
    device = torch.device("cpu")
    net.to(device)

    return net, costs

def fit_reg(x, y, theta, max_iter=500, alpha=0.2, batch_size=-1):
    """
    Train a logistic regression model
    """
    global classes
    thetas = []
    costs = np.zeros(max_iter)
    grad = np.zeros((max_iter, x.shape[1]))

    num_batches = 1
    if batch_size > 0:
        num_batches = int(x.shape[0]/batch_size)

    # Train for max_iter epochs
    for epoch in range(max_iter):
        indices = torch.randperm(x.shape[0])
        indices = np.array(indices)
        # Iterate over each mini-batch
        for b in range(num_batches):
            batch_indices = indices[b*batch_size: (b+1)*batch_size]
            batch_X = x[batch_indices]
            batch_Y = y[batch_indices]
            # Calculate loss and update model
            costs[epoch], grad_tmp = cost_reg(theta, batch_X, batch_Y)
            theta += alpha * grad_tmp
        costs[epoch], grad[epoch] = cost_reg(theta, x, y)
    return theta, grad, costs

def accuracy_reg(theta, x, y):
    """
    Calculate accuracy of regression model
    """
    preds = [sigmoid(xi @ theta) for xi in x]
    pred_y = np.round(preds)

    # one vs. rest binary classification
    conditions = False
    for c in classes[0:int(len(classes)/2)]:
        conditions = conditions | (y == c)
    binary_y = np.where(conditions, 1, 0)

    incorrect = np.count_nonzero(binary_y-pred_y)
    return 1-incorrect/y.shape[0]

def accuracy_nn(model, x, y):
    """
    Calculate accuracy of Neural Network model
    """
    global BATCH_SIZE

    # Transfer data to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    incorrect = 0
    num_batches = int(x.shape[0]/BATCH_SIZE)

    # Iterate over all batches
    for b in range(num_batches):
        batch_indices = list(range(b*BATCH_SIZE, (b+1)*BATCH_SIZE))
        batch_X = x[batch_indices]
        batch_Y = y[batch_indices]
        
        if torch.cuda.is_available():
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()

        pred_y = model(batch_X)
        incorrect += torch.nonzero(batch_Y-torch.argmax(pred_y,axis=1)).size(0)

    # Move data back to CPU
    device = torch.device("cpu")
    model.to(device)
    return 1-incorrect/y.shape[0]

def average_regs(model, client_models, fracs):
    """
    Average regression model with other client models
    """
    num_clients = len(client_models)
    theta_sum = np.zeros(model.shape[0])
    i = 0
    for theta in client_models:
        theta_sum += theta*fracs[i]
        i += 1
    model = theta_sum #/ num_clients
    return model

def average_nns(model, client_models, fracs):
    """
    Average Neural Net with other client models
    """
    num_clients = len(client_models)
    params = []
    for param in model.parameters():
        params.append(0)
    j = 0
    for net in client_models:
        i = 0
        for param in net.parameters():
            params[i] += param.data*fracs[j] 
            i += 1
        j += 1
    i = 0
    for param in model.parameters():
        param.data = params[i]#/num_clients 
        i += 1
    return model

def client(num, offset, model, epochs, X, Y, model_type):
    """
    Train client with SGD and random subset of data
    """
    global BATCH_SIZE
    # Make IID
    indices = torch.randperm(X.shape[0])
    indices = np.array(indices)
    batch_indices = indices[0:offset]
    client_X = X[batch_indices]
    client_Y = Y[batch_indices]

    if model_type == 0: # Regression model
        theta, grad, costs = fit_reg(client_X, client_Y, model, 
                max_iter=epochs, batch_size=BATCH_SIZE)
        return theta, costs
    else: # Any NN model
        model, costs = fit_nn(client_X, client_Y, model, 
                max_iter=epochs, batch_size=BATCH_SIZE)
        return model, costs
    
def get_client_epochs(local_epochs, prob, n):
    """
    Random decisions to take a local step or not
    """
    client_epochs = local_epochs
    if prob == 1:
        global chance
        client_epochs = 0
        for p in range(0,local_epochs):
            client_epochs += random.random() < chance
    if prob == 2:
        client_epochs = 0
        for p in range(0,local_epochs):
            client_epochs += random.random() < 0.1*(n+1) 
    if prob == 3:
        client_epochs = 0
        if n == 0:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.1 
        else:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.6 
    if prob == 4:
        client_epochs = 0
        if n == 0:
            client_epochs = local_epochs 
        else:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.5 
    if prob == 5:
        client_epochs = 0
        if n == 0:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.6 
        else:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.9 
    return client_epochs

def initializeNN(model_type):
    """
    Intialize NN Model
    """
    model = None
    if model_type == 1:
        model = cNet()
    if model_type == 2:
        model = CIFARNet() 
    if model_type == 3:
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False) 

    # Move to CPU
    device = torch.device("cpu")
    model.to(device)
    return model

def federated_sets(num_centers, num_clients):
    """
    Set different sized dataset for clients and hubs
    """
    offset_hub = []
    offset_client = []
    if num_centers == 1:
        offset_hub = [1]
        for i in range(num_clients):
            if i < 20:
                offset_client.append(0.0025)
            elif i < 40:
                offset_client.append(0.005)
            elif i < 60:
                offset_client.append(0.01)
            elif i < 80:
                offset_client.append(0.0125)
            else:
                offset_client.append(0.02)
    # MLL-SGD
    else:
        for i in range(num_clients):
            if i < 2:
                offset_client.append(0.025)
                offset_hub.append(0.025)
            elif i < 4:
                offset_client.append(0.05)
                offset_hub.append(0.05)
            elif i < 6:
                offset_client.append(0.1)
                offset_hub.append(0.1)
            elif i < 8:
                offset_client.append(0.125)
                offset_hub.append(0.125)
            else:
                offset_client.append(0.2)
                offset_hub.append(0.2)
    return offset_hub, offset_client

def fed_hierarchy(x, y, xtest, ytest, model_type=0, num_centers=1, 
                    num_clients=1, epochs=500, local_epochs=1, 
                    A=None, center_epochs=1, prob=0, 
                    dataset=0, graph=5, fed=False):
    """
    Train in Multi-Level Local SGD
    """
    global chance, learning_rate

    # Setting up filename based on type of experiment
    chancestr = ""
    if chance != 0.55:
        chancestr = str(chance)     

    # Arrays for averaging models
    offset_hub = [1 / num_centers] * num_centers
    offset_client = [1 / num_clients] * (num_clients)
    W = np.ones((num_centers,num_centers))*(1 / num_centers)

    fedname = ""
    # If we are running an experiment that uses uneven data distribution
    if fed:
        fedname = "fed"
        offset_hub, offset_client = federated_sets(num_centers, num_clients)

    if A is None:
        A = np.ones((num_centers,num_centers), dtype=int)
    else:
        # If hub network is not complete, calculate diffusion matrix
        Anew = np.array(A)-np.eye(num_centers)
        d = np.sum(Anew,axis=0)
        D = np.diag(d)
        alpha = 1/(np.max(d)*50+1)
        weights = np.linalg.inv(np.diag(offset_hub))
        W = np.eye(num_centers) - alpha*weights.dot(D - Anew)

    # Determine which model to train
    model = None
    cost = None
    average_models = None
    accuracy = None
    if model_type == 0:
        model = np.zeros(x.shape[1])
        cost = cost_reg
        average_models = average_regs
        accuracy = accuracy_reg
    else:
        model = initializeNN(model_type)
        cost = cost_nn
        average_models = average_nns
        accuracy = accuracy_nn
        if model_type == 3:
            learning_rate = 0.1

    # Copy initial model for all hubs
    models = []
    for n in range(0, num_centers):
        models.append(copy.deepcopy(model))

    costs = np.zeros(epochs)
    success_rate = np.zeros(epochs)
    clients_run =  np.zeros((epochs*center_epochs, num_centers, num_clients))

    # Train for epochs*center_epochs iterations
    for e in tqdm(range(0, epochs*center_epochs)):
        # If using ResNet-18, adjust learning rate every third of training
        if model_type == 3 and e > 2*epochs*center_epochs/3:
            learning_rate = 0.001
        elif model_type == 3 and e > epochs*center_epochs/3:
            learning_rate = 0.01

        # Iterate over each hub
        for n in range(0, num_centers):
            client_models = [None]*num_clients
            # Train on each client
            for i in range(0, num_clients):
                client_epochs = get_client_epochs(local_epochs,prob,i)
                clients_run[e, n, i] = client_epochs
                # Run local training for client i
                offset = int(offset_client[i]*y.shape[0]/num_centers)
                (client_models[i], cost_val) = client(
                            n*num_clients+i, 
                            offset, 
                            copy.deepcopy(models[n]), 
                            client_epochs, x, y, model_type) 

            # Average models in sub-network
            models[n] = average_models(models[n],client_models,
                    np.array(offset_client))
                
        # Each hub average with neighbors every center_epochs
        if e % center_epochs == 0:
            for i in range(0, num_centers):
                # Center i average with center j
                neighbor_models = [models[j] for j in range(num_centers) if(A[i][j]==1)]
                offset_neighbor = [offset_hub[j] for j in range(num_centers) if(A[i][j]==1)]
                models[i] = average_models(models[i], neighbor_models, 
                        np.array(offset_neighbor)/np.sum(offset_neighbor))
                #loss,_ = cost(models[i], x, y)
                #print("Model "+str(i)+" Loss: "+str(loss))
        
            # Calculate the average model in the system
            avg_model = average_models(models[0], models, np.array(offset_hub))
            costs[int(e/center_epochs)], _ = cost(avg_model, x, y)
            print("Weighted Average Model Loss: "+str(costs[int(e/center_epochs)]))
            #avg_model = average_models(models[0], models, 
            #        np.array([[1 / num_centers] * num_centers]), 0)
            #loss, _ = cost(avg_model, x, y)
            #print("Average Model Loss: "+str(loss))

            # Save results
            success_rate[int(e/center_epochs)] = accuracy(avg_model, xtest, ytest)
            filename = f"results/loss_model{model_type}_data{dataset}_hubs{num_centers}_workers{num_clients}_tau{local_epochs}_q{center_epochs}_graph{graph}_prob{prob}{fedname}{chancestr}.p"
            pickle.dump(costs, open(filename,'wb'))
            filename = f"results/accuracy_model{model_type}_data{dataset}_hubs{num_centers}_workers{num_clients}_tau{local_epochs}_q{center_epochs}_graph{graph}_prob{prob}{fedname}{chancestr}.p"
            pickle.dump(success_rate, open(filename,'wb'))
            filename = f"results/coin_model{model_type}_data{dataset}_hubs{num_centers}_workers{num_clients}_tau{local_epochs}_q{center_epochs}_graph{graph}_prob{prob}{chancestr}.p"
            pickle.dump(clients_run, open(filename,'wb'))
    return costs,success_rate

def load_graph(g,num_centers):
    """
    Load in adjacency matrix from graph file with ID 'g'
    """
    A = []
    if g == 6:
        for i in range(num_centers):
            A.append([])
            connect_left = i-1
            if connect_left < 0:
                conenct_left = num_centers-1
            connect_right = i+1
            if connect_left >= num_centers:
                conenct_left = 0
            for j in range(num_centers):
                if i == j or j == connect_left or j == connect_right:
                    A[i].append(1)
                else:
                    A[i].append(0)
    else:
        graph_file = open("graphs/graph_"+str(g)+".txt",'r')
        lines = graph_file.readlines()
        i = 0
        for line in lines:
            row = line.split(' ')
            A.append([])
            for r in row:
                A[i].append(int(r))
            i += 1

    # Calculate average degree and create degree vector 
    average_degree = 0
    d = []
    for i in range(0, len(A)):
        degree = 0
        for j in range(0, len(A)):
            if i != j and A[i][j] == 1:
                degree += 1
            #A[j][i] = A[i][j]
        d.append(degree)
        average_degree += degree
    num_centers = len(A[0])
    average_degree /= num_centers

    # Find second largest eigenvalue of mixing matrix
    Anew = np.array(A)-np.eye(num_centers)
    D = np.diag(d)
    alpha = 1/(np.max(d)+1)
    W = np.eye(num_centers) - alpha*(D - Anew)
    eigvals, eigvecs = np.linalg.eig(W)
    np.apply_along_axis(abs,0,eigvals)
    eigvals.sort()

    return A,eigvals[-2]

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Run Multi-Level Local SGD.')
    parser.add_argument('--data', type=int, nargs='?', default=0,
                            help='dataset to use in training.')
    parser.add_argument('--model', type=int, nargs='?', default=0,
                            help='model to use in training.')
    parser.add_argument('--hubs', type=int, nargs='?', default=1,
                            help='number of hubs in system.')
    parser.add_argument('--workers', type=int, nargs='?', default=1,
                            help='number of workers per hub.')
    parser.add_argument('--tau', type=int, nargs='?', default=1,
                            help='number of local iterations for worker.')
    parser.add_argument('--q', type=int, nargs='?', default=1,
                            help='number of sub-network iterations before global averaging.')
    parser.add_argument('--graph', type=int, nargs='?', default=0,
                            help='graph file ID to use for hub network.')
    parser.add_argument('--epochs', type=int, nargs='?', default=1000,
                            help='Number of epochs/global iterations to train for.')
    parser.add_argument('--batch', type=int, nargs='?', default=1000,
                            help='Batch size to use in Mini-batch SGD.')
    parser.add_argument('--prob', type=int, nargs='?', default=0,
                            help='Indicates with probability distribution to use for workers.')
    parser.add_argument('--fed', type=bool, nargs='?', default=False,
                            help='Indicates if worker sets should be different sizes.')
    parser.add_argument('--chance', type=float, nargs='?', default=0.55,
                            help='Fixed probability of taking gradient step.')

    args = parser.parse_args()
    print(args)
    return args

def main():
    global classes, BATCH_SIZE, chance
    
    # Parse input arguments
    args = parse_args()
    A = None
    hubs = args.hubs
    if args.graph != 5:
        A,sparsity = load_graph(args.graph,hubs)
    dataset = args.data
    model_type = args.model
    q = args.q
    tau = args.tau
    workers = args.workers
    max_iter = args.epochs 
    BATCH_SIZE = args.batch
    prob = args.prob
    fed = args.fed
    chance = args.chance

    # Set up filename based on which experiment is running
    fedname = ""
    if fed:
        fedname = "fed"

    if(dataset == 0):
        # Load MNIST data
        mndata = MNIST('./MNIST/raw/')
        X, y = mndata.load_training()
        Xtest, ytest = mndata.load_testing()

    if(dataset == 1):
        # Load EMNIST data
        X, y = extract_training_samples('letters')
        Xtest, ytest = extract_test_samples('letters')
        X = X.reshape(-1, X.shape[1]*X.shape[2])
        Xtest = Xtest.reshape(-1, Xtest.shape[1]*Xtest.shape[2])

    if(dataset == 2):
        # Load CIFAR data
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
        X = trainset.data
        y = trainset.targets
        Xtest = testset.data
        ytest = testset.targets


    classes = np.unique(y)
    if dataset != 2:
        # Normalize MNIST and EMNIST data
        mm_scaler = MinMaxScaler()
        X = mm_scaler.fit_transform(X)
        Xtest = mm_scaler.fit_transform(Xtest)

    model = None
    if(model_type == 0):
        # Load data for regression model
        X = np.array(X)
        y = np.array(y)
        X = np.insert(X, 0, 1, axis=1)
        Xtest = np.array(Xtest)
        ytest = np.array(ytest)
        Xtest = np.insert(Xtest, 0, 1, axis=1)
        model = np.zeros(X.shape[1])

    if(model_type == 1):
        # Load data for CNN 
        X = torch.tensor(X, dtype=torch.float32)
        X = X.reshape(-1,28,28)
        X.unsqueeze_(-1)
        X = X.transpose(3,1)
        y = torch.tensor(y, dtype=torch.long)
        Xtest = torch.tensor(Xtest, dtype=torch.float32)
        Xtest = Xtest.reshape(-1,28,28)
        Xtest.unsqueeze_(-1)
        Xtest = Xtest.transpose(3,1)
        ytest = torch.tensor(ytest, dtype=torch.long)

    if(model_type == 2 or model_type == 3):
        # Load data for CIFAR-10 data used with CIFARNet or ResNet-18 
        X = torch.tensor(X, dtype=torch.float32)
        X = X.reshape(-1,3,32,32)
        #X.unsqueeze_(-1)
        #X = X.transpose(3,1)
        y = torch.tensor(y, dtype=torch.long)
        Xtest = torch.tensor(Xtest, dtype=torch.float32)
        Xtest = Xtest.reshape(-1,3,32,32)
        #Xtest.unsqueeze_(-1)
        #Xtest = Xtest.transpose(3,1)
        ytest = torch.tensor(ytest, dtype=torch.long)

    # Run main training loop
    costs,success = fed_hierarchy(X, y, Xtest, ytest,
                model_type=model_type, num_centers=hubs, num_clients=workers,
                epochs=max_iter, local_epochs=tau, center_epochs=q, 
                A=A, prob=prob, dataset=dataset, graph=args.graph,
                fed=fed)

    # Save results
    filename = f"results/loss_model{model_type}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}.p"
    pickle.dump(costs, open(filename,'wb'))
    filename = f"results/accuracy_model{model_type}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}.p"
    pickle.dump(success, open(filename,'wb'))
    print(costs)
    print(success)


if __name__ == "__main__":
    main()
