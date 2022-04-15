import os
import random

import torch
from torch import nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        torch.nn.init.normal(self.fc.weight, mean=0., std=0.05)
        torch.nn.init.zeros_(self.fc.bias)

        '''
        torch.nn.init.normal(self.gru.all_weights[0], mean=0., std=0.05)
        torch.nn.init.normal(self.gru.all_weights[1], mean=0., std=0.05)
        torch.nn.init.normal(self.gru.all_weights[2], mean=0., std=0.05)
        torch.nn.init.zeros_(self.gru.bias)
        '''
        
    
    def forward(self, x):
        hidden_state = self.init_hidden()
        output, hidden_state = self.gru(x, hidden_state)
        output = self.fc(output[-1])
        return output
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
    
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(hidden_size, output_size)
        
        torch.nn.init.normal(self.in2hidden.weight, mean=0., std=0.05)
        torch.nn.init.normal(self.in2output.weight, mean=0., std=0.02)
        #nn.init.kaiming_uniform_(self.in2hidden.weight)
        #nn.init.kaiming_uniform_(self.in2output.weight)
        
        torch.nn.init.zeros_(self.in2hidden.bias)
        torch.nn.init.zeros_(self.in2output.bias)  
    
    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(hidden)
        return output, hidden
    
    def init_hidden(self):
        # return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
        return torch.zeros(1, self.hidden_size)    