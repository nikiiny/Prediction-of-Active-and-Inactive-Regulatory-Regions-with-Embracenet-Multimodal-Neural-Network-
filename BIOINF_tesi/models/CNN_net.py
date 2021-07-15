import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from .utils import size_out_convolution


class CNN_define_model(nn.Module):

    def __init__(self, trial, classes=2):
        super(CNN_define_model, self).__init__()
        self.trial = trial
        self.classes = classes
        self.model = []
        
        maxpool_kernel_size=10
        stride=1
        maxpool_stride=2
        input_size=256
        in_channels = 1
        
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        
        for i in range(n_layers):
            if i==0:
                out_channels = trial.suggest_categorical("out_channels_l{}".format(i), [16, 32, 64])
            elif i==1:
                out_channels = trial.suggest_categorical("out_channels_l{}".format(i), [32, 64, 96])
            elif i==2:
                out_channels = trial.suggest_categorical("out_channels_l{}".format(i), [64, 96, 128, 256])
            
            kernel_size = trial.suggest_categorical("kernel_size_l{}".format(i), [5, 11, 15])
            padding = (kernel_size-1)/2 # same padding
            layers.append( nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                     padding=padding )) 
            layers.append( nn.BatchNorm1d(out_channels) )
            layers.append( nn.ReLU() )
                          
            layers.append( nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride) )

            if i==0:
                layers.append(nn.Dropout(0.3)) 
            else:
                layers.append(nn.Dropout(0.4))

            in_channels = out_channels
            
            # calculate size of FC layer
            # for convolution 
            output_size = size_out_convolution(input_size, kernel_size, padding, stride)
            # for maxpool 
            output_size = size_out_convolution(output_size, maxpool_kernel_size, padding=0, maxpool_stride)
            input_size=output_size

        out = out.reshape(out.size(0), -1) 
        
        # to calculate the size of the FC layer:
        # - calculate inpute size after each convolution
        # - the input to FC layer is going to be num channels x length
        
        fc_layer_size = output_size*out_channels
        
        
        layers.append( nn.Linear(fc_layer_size, 1000) )
        layers.append( nn.Linear(fc_layer_size, 64) )
        layers.append( nn.Linear(64, self.classes) )

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.model(x)