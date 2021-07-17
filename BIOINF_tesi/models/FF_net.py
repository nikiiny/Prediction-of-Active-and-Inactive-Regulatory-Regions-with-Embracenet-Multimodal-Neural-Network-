import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class FFNN_define_model(nn.Module):

    def __init__(self, trial, in_features, classes=2):
        super(FFNN_define_model, self).__init__()
        self.trial = trial
        self.classes = classes
        self.model = []
        
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = self.trial.suggest_int("n_layers", 1, 3)
        layers = []

        for i in range(n_layers):
            if i==0:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [4, 16, 32, 64, 128, 256])
            elif i==1:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [4, 16, 32, 64, 128])
            elif i==2:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [4, 16, 32, 64])
                
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            
            if i==0:
                layers.append(nn.Dropout(0.3))
            else:
                layers.append(nn.Dropout(0.5))
                
                
            in_features = out_features

        layers.append(nn.Linear(in_features, self.classes))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.model(x)