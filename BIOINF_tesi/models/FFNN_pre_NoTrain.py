import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class FFNN_pre_NoTrain(nn.Module): 

    def __init__(self, in_features, model_params, device):
        super(FFNN_pre_NoTrain, self).__init__()
        self.device = device
        self.model = []

        n_layers = model_params['n_layers']
        layers = []

        for i in range(n_layers):

            out_features = model_params[f'n_units_l{i}']

            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_params[f'dropout_l{i}']))
                
            in_features = out_features

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        
        out = self.model(x)
        #out = out.reshape(out.size(0), -1)  #?

        return out.to(self.device)
