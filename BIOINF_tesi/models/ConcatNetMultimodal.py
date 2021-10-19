import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .utils import output_size_from_model_params, drop_last_layers
from . import CNN_pre, FFNN_pre



class ConcatNetMultimodal(nn.Module):
    def __init__(self, 
                 trial,
                 cell_line,
                 task,
                 device,
                 in_features_FFNN,
                 n_classes=2,  
                 args=None,
                 embracenet_dropout=True):
        super(ConcatNetMultimodal, self).__init__()
        
        
        # input parameters
        self.trial=trial
        self.cell_line=cell_line
        self.device = device
        self.n_classes = n_classes
        self.args = args
        
        # 1) pre neural networks
        self.FFNN = FFNN_pre(self.trial, in_features_FFNN, device=self.device) #?
        self.CNN = CNN_pre(self.trial, device=self.device) #?
        
        self.FFNN_pre_output_size = self.FFNN.output_size
        self.CNN_pre_output_size = self.CNN.output_size

        
        # 2) concatenation layer + post layers
        in_features = self.FFNN_pre_output_size + self.CNN_pre_output_size

        n_post_layers = self.trial.suggest_int("CONCATNET_n_post_layers", 1, 3) 
        post_layers = []

        for i in range(n_post_layers):
            if i==0:
                out_features = self.trial.suggest_categorical("CONCATNET_n_units_l{}".format(i), [512, 768, 1024])
            elif i==1:
                out_features = self.trial.suggest_categorical("CONCATNET_n_units_l{}".format(i), [32, 64, 128, 256, 512])
            elif i==2:
                out_features = self.trial.suggest_categorical("CONCATNET_n_units_l{}".format(i), [16, 32, 64, 128, 256])
                
            post_layers.append(nn.Linear(in_features, out_features))
            post_layers.append(nn.ReLU())
            
            dropout = self.trial.suggest_categorical("CONCATNET_dropout_l{}".format(i), [0.0, 0.2, 0.3, 0.5])
            post_layers.append(nn.Dropout(dropout))
                
            in_features = out_features
            
        post_layers.append(nn.Linear(in_features, self.n_classes))
        # n of layers to tune

        self.post = nn.Sequential(*post_layers)
  
    def forward(self, x):
    # availabilities: don't change since we have all the data
    # selection probabilities: weight the contribution according to the auprc
    
        x_FFNN, x_CNN = x
        x_FFNN = self.FFNN(x_FFNN)
        x_CNN = self.CNN(x_CNN)

        # concat layer
        output = torch.cat((x_FFNN, x_CNN), dim=1)

        # employ final layers
        output = self.post(output)

        # output softmax
        return output

