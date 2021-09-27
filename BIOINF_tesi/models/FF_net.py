import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class FFNN(nn.Module): 

    def __init__(self, trial, in_features, device, classes=2):
        super(FFNN, self).__init__()
        self.trial = trial
        self.classes = classes
        self.device = device
        self.model = []
        
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = self.trial.suggest_int("n_layers", 1, 4)
        layers = []

        for i in range(n_layers):
            if i==0:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [32, 64, 128, 256])
            elif i==1:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [16, 32, 64, 128])
            elif i==2:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [4, 16, 32, 64])
            elif i==3:
                out_features = self.trial.suggest_categorical("n_units_l{}".format(i), [4, 16, 32])
                
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            
            if i<2:
                dropout = self.trial.suggest_categorical("dropout_l{}".format(i), [0, 0.2, 0.3, 0.4])
                layers.append(nn.Dropout(dropout))
            elif i>=2:
                dropout = self.trial.suggest_categorical("dropout_l{}".format(i), [0, 0.4, 0.5])
                layers.append(nn.Dropout(dropout))
                
                
            in_features = out_features

        layers.append(nn.Linear(in_features, self.classes))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.model(x).to(self.device)