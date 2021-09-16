import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from .utils import size_out_convolution 


class CNN_LSTM(nn.Module):

    def __init__(self, trial, device, classes=2):
        super(CNN_LSTM, self).__init__()
        self.trial = trial
        self.classes = classes
        self.device = device
        self.CNN_model = []
        self.LSTM_model = []
        
        maxpool_kernel_size=10
        stride=1
        maxpool_stride=2
        input_size=256
        in_channels = 4
        
        n_layers = trial.suggest_int("CNN_n_layers", 1, 2)
        layers = []
        
        for i in range(n_layers):
            if i==0:
                out_channels = trial.suggest_categorical("CNN_out_channels_l{}".format(i), [16, 32, 64])
            elif i==1:
                out_channels = trial.suggest_categorical("CNN_out_channels_l{}".format(i), [32, 64, 96])
            
            kernel_size = trial.suggest_categorical("kernel_size_l{}".format(i), [5, 11, 15])
            padding = int((kernel_size-1)/2) # same padding
            layers.append( nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                     padding=padding )) 
            layers.append( nn.BatchNorm1d(out_channels) )
            layers.append( nn.ReLU() )
                          
            layers.append( nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride) )

            if i<2:
                dropout = self.trial.suggest_categorical("dropout_l{}".format(i), [0, 0.3, 0.4])
                layers.append(nn.Dropout(dropout))
            elif i==2:
                dropout = self.trial.suggest_categorical("dropout_l{}".format(i), [0, 0.4, 0.5])
                layers.append(nn.Dropout(dropout))

            in_channels = out_channels
            
            # calculate size of FC layer
            # for convolution 
            output_size = size_out_convolution(input_size, kernel_size, padding, stride)
            # for maxpool 
            output_size = size_out_convolution(output_size, maxpool_kernel_size, 0, maxpool_stride)
            input_size=output_size # length of sequence after convolution

       
        self.CNN_model = nn.Sequential(*layers)

        
        hidden_layer_size = trial.suggest_categorical("LSTM_hidden_layer_size", [32, 64, 128])
        n_layers = trial.suggest_int("LSTM_n_layers", 1,2)

        
        # calcola output cnn come prima. lunghezza sequenza è seq_len
        
        self.LSTM_model = nn.LSTM(4, hidden_layer_size, n_layers, batch_first=True) 
        
        self.seq_len = output_size
        
        self.last_layer2 = nn.Linear(1000, 64) 
        self.last_output = nn.Linear(64, self.classes) 

        
    
    def forward(self, x):
        
        out = self.CNN_model(x).to(self.device)
        out = out.reshape(out.size(0),-1, 4) # batch, timesteps, rest #why????????
        out, _ = self.LSTM_model(out)
        out = out.reshape(out.size(0),-1) # batch_size, rest
        self.last_layer1 = nn.Linear(out.size(1), 1000).to(self.device)
        out = self.last_layer1.float()(out.float())
        out = self.last_layer2.float()(out)
        out = self.last_output.float()(out)
    
        # INPUT SIZE of LSTM: n_features
        # INPUT of LSTM: batch_size, seq_len, input_size (100, 256, 4) #?
        # OUTPUT HAS SHAPE (batch_size, seq_len, hidden_dim)
        # seq len è l output del cnn
        
        return out.to(self.device)