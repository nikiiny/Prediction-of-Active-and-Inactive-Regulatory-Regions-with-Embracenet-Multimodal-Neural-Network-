import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class FFNN_NoTrain(nn.Module): 

    def __init__(self, 
                cell_line,
                task,
                in_features, 
                torch_saved_state, 
                device, 
                classes=2):
        super(FFNN_NoTrain, self).__init__()
        self.cell_line = cell_line
        self.task = task
        self.device = device
        self.classes = classes
        self.model = []
        self.softmax_layer = torch.nn.Softmax(dim=None)

        torch_saved_state = torch.load(f'models/{self.cell_line}_{self.task}_FFNN_TEST_augmentation.pt', 
            map_location=torch.device(device))

        model_params = torch_saved_state['model_params']
        n_layers = model_params['n_layers']
        layers = []

        for i in range(n_layers):

            out_features = model_params[f'n_units_l{i}']

            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_params[f'dropout_l{i}']))
                
            in_features = out_features

        layers.append(nn.Linear(in_features, self.classes))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        
        out = self.model(x)
        out = self.softmax_layer(out)

        return out.reshape(-1).to(self.device)
