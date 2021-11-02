import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from .utils import size_out_convolution

import re

class CNN_NoTrain(nn.Module): 

    def __init__(self, 
                cell_line,
                task,
                torch_saved_state, 
                device, 
                classes=2):
        super(CNN_NoTrain, self).__init__()
        self.cell_line = cell_line
        self.task = task
        self.device = device
        self.classes = classes
        self.softmax_layer = torch.nn.Softmax(dim=None)
        self.CNN_model = []
        
        maxpool_kernel_size=10
        stride=1
        maxpool_stride=2
        input_size=256
        in_channels = 4
        
        torch_saved_state = torch.load(f'models/{self.cell_line}_{self.task}_FFNN_TEST_augmentation.pt', 
            map_location=torch.device(device))

        model_params = torch_saved_state['model_params']
        n_layers = model_params['n_layers']

        layers = []
        
        for i in range(n_layers):
        
            kernel_size = model_params[f'kernel_size_l{i}']

            out_channels = model_params[f'out_channels_l{i}']

            padding = int((kernel_size-1)/2) # same padding
            layers.append( nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                     padding=padding ))
            layers.append( nn.BatchNorm1d(out_channels) )
            layers.append( nn.ReLU() )
                          
            layers.append( nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride) )

            layers.append(nn.Dropout(model_params[f'dropout_l{i}']))

            in_channels = out_channels
            output_size = size_out_convolution(input_size, kernel_size, padding, stride)
            # for maxpool 
            output_size = size_out_convolution(output_size, maxpool_kernel_size, 0, maxpool_stride)
            input_size = output_size

        self.CNN_model = nn.Sequential(*layers)
        self.last_layer1 = nn.Linear(out_channels*output_size, 1000) 
        self.last_layer2 = nn.Linear(1000, 64) 
        self.last_output = nn.Linear(64, self.classes) 
    
    def forward(self, x):

        out = self.CNN_model(x)
        out = out.reshape(out.size(0), -1) # batch_size, rest
        out = self.last_layer1(out)
        out = self.last_layer2(out)
        out = self.last_output(out)
        out = self.softmax_layer(out)
    
        return out.reshape(-1).to(self.device)