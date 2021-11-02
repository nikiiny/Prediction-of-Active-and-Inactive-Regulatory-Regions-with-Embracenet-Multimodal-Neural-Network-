import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .utils import output_size_from_model_params, drop_last_layers, get_single_model_params
from . import FFNN_pre_NoTrain, CNN_pre_NoTrain


class ConcatNetMultimodal_NoTrain(nn.Module):
    def __init__(self, 
                 cell_line,
                 task,
                 in_features_FFNN,
                 device,
                 n_classes=2,  
                 args=None):
        super(ConcatNetMultimodal_NoTrain, self).__init__()
        
        
        # input parameters
        self.cell_line=cell_line
        self.task = task
        self.device = device
        self.n_classes = n_classes
        self.softmax_layer = torch.nn.Softmax(dim=None)
        self.args = args
        
        torch_saved_state = torch.load(f'models/{self.cell_line}_{self.task}_ConcatNetMultimodal_TEST.pt', 
            map_location=torch.device(device))

        single_model_params = get_single_model_params(torch_saved_state['model_params'])
        # 1) pre neural networks
        self.FFNN = FFNN_pre_NoTrain(in_features_FFNN, single_model_params['FFNN'], device=self.device)
        self.CNN = CNN_pre_NoTrain(single_model_params['CNN'], device=self.device) 
        
        for param in self.FFNN.parameters():
            param.requires_grad = False
        for param in self.CNN.parameters():
            param.requires_grad = False
        
        last_layer_FFNN = single_model_params['FFNN']['n_layers']-1
        self.FFNN_pre_output_size = single_model_params['FFNN'][f'n_units_l{last_layer_FFNN}']
        
        self.CNN_pre_output_size = output_size_from_model_params(single_model_params['CNN'])
        
        # 2) concatenation layer + post layers
        model_params = torch_saved_state['model_params']

        in_features = self.FFNN_pre_output_size + self.CNN_pre_output_size

        n_post_layers = model_params["CONCATNET_n_post_layers"]
        post_layers = []

        for i in range(n_post_layers):
            out_features = model_params[f"CONCATNET_n_units_l{i}"]
                
            post_layers.append(nn.Linear(in_features, out_features))
            post_layers.append(nn.ReLU())
            
            dropout = model_params[f"CONCATNET_dropout_l{i}"]
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
        outuput = self.softmax_layer(output)

        # output softmax
        return output.reshape(-1)

