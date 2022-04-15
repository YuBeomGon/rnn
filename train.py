import os
import random
import torch
from torch import nn
import torch.nn.functional as F

from config import *

def evaluate(model, device, test_dataset) : 
    
    num_correct = 0
    num_samples = len(test_dataset)    

    model.eval()

    with torch.no_grad():
        for name, label in test_dataset:
            name = name.to(device)
            label = label.to(device)            
            output = model(name)
            _, pred = torch.max(output, dim=1)
            num_correct += bool(pred == label)
            
    model.train()

    print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")
    return (num_correct / num_samples * 100)
    
def evaluate_rnn(model, device, test_dataset) :    
    
    num_correct = 0
    num_samples = len(test_dataset)

    model.eval()

    with torch.no_grad():
        for name, label in test_dataset:
            name = name.to(device)
            label = label.to(device)            
            hidden_state = model.init_hidden()
            hidden_state = hidden_state.to(device)        

            for char in name:
                output, hidden_state = model(char, hidden_state)

            _, pred = torch.max(output, dim=1)
            num_correct += bool(pred == label)
            
    model.train()

    print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")    
    return (num_correct / num_samples * 100)
    
    
def train(model, device, train_dataset, test_dataset, optimizer, criterion) :
    
    model.train()
    acc_list = []
    
    for epoch in range(num_epochs):
        random.shuffle(train_dataset)
        losses = []
        for i, (name, label) in enumerate(train_dataset):
            name = name.to(device)
            label = label.to(device)
            output = model(name)
            loss = criterion(output, label)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_dataset)}], "
                    f"Loss: {sum(losses)/print_interval:.4f}"
                )
                acc = evaluate(model, device, test_dataset)    
                acc_list.append(acc)
                losses = []
    return acc_list
                
def train_rnn(model, device, train_dataset, test_dataset, optimizer, criterion) :      
    
    model.train()
    acc_list = []     
    
    for epoch in range(num_epochs):
        random.shuffle(train_dataset)
        losses = []
        for i, (name, label) in enumerate(train_dataset):
            name = name.to(device)
            label = label.to(device)        
            hidden_state = model.init_hidden()
            hidden_state = hidden_state.to(device)

            for char in name:
                output, hidden_state = model(char, hidden_state)
            loss = criterion(output, label)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_dataset)}], "
                    f"Loss: {sum(losses)/print_interval:.4f}"
                )              
                acc = evaluate_rnn(model, device, test_dataset)    
                acc_list.append(acc)
                losses = []
    return acc_list                