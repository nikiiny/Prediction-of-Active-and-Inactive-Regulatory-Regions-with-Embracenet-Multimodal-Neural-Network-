import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch.optim as optim
from .utils import size_out_convolution


class CNN_pre(nn.Module): 

    def __init__(self, trial, device):
        super(CNN_pre, self).__init__()
        self.trial = trial
        self.device = device
        self.CNN_model = []
        
        maxpool_kernel_size=10
        stride=1
        maxpool_stride=2
        input_size=256
        in_channels = 4
        
        n_layers = trial.suggest_int("CNN_n_layers", 1, 4)
        layers = []
        
        for i in range(n_layers):
            if i==0:
                out_channels = trial.suggest_categorical("CNN_out_channels_l{}".format(i), [16, 32, 64])
            elif i==1:
                out_channels = trial.suggest_categorical("CNN_out_channels_l{}".format(i), [32, 64, 96])
            elif i==2:
                out_channels = trial.suggest_categorical("CNN_out_channels_l{}".format(i), [64, 96, 128, 256])
            elif i==3:
                out_channels = trial.suggest_categorical("CNN_out_channels_l{}".format(i), [128, 256, 512])
            
            kernel_size = trial.suggest_categorical("CNN_kernel_size_l{}".format(i), [5, 11, 15])
            padding = int((kernel_size-1)/2) # same padding
            layers.append( nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                     padding=padding )) 
            layers.append( nn.BatchNorm1d(out_channels) )
            layers.append( nn.ReLU() )
                          
            layers.append( nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride) )

            if i<1:
                dropout = self.trial.suggest_categorical("CNN_dropout_l{}".format(i), [0, 0.2, 0.3, 0.4])
                layers.append(nn.Dropout(dropout))
            elif i>=1:
                dropout = self.trial.suggest_categorical("CNN_dropout_l{}".format(i), [0, 0.4, 0.5])
                layers.append(nn.Dropout(dropout))

            in_channels = out_channels
            
            # calculate size of FC layer
            # for convolution 
            output_size = size_out_convolution(input_size, kernel_size, padding, stride)
            # for maxpool 
            output_size = size_out_convolution(output_size, maxpool_kernel_size, 0, maxpool_stride)
            input_size=output_size

        # out = out.reshape(out.size(0), -1) # batch_size, rest
        
        # to calculate the size of the FC layer:
        # - calculate inpute size after each convolution
        # - the input to FC layer is going to be num channels x length

        self.output_size = out_channels*output_size
        self.CNN_model = nn.Sequential(*layers)
    
    def forward(self, x):

        out = self.CNN_model(x)
        out = out.reshape(out.size(0), -1) # batch_size, rest

        return out.to(self.device)