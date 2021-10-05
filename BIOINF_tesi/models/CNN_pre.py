import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from .utils import size_out_convolution

import re

class CNN_pre(nn.Module): 

    def __init__(self, torch_saved_state, device):
        super(CNN_pre, self).__init__()
        self.device = device
        self.CNN_model = []
        
        maxpool_kernel_size=10
        stride=1
        maxpool_stride=2
        input_size=256
        in_channels = 4
        
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
            

        self.CNN_model = nn.Sequential(*layers)

    
    def forward(self, x):

        out = self.CNN_model(x)
        out = out.reshape(out.size(0), -1) # batch_size, rest
    
        return out.to(self.device)