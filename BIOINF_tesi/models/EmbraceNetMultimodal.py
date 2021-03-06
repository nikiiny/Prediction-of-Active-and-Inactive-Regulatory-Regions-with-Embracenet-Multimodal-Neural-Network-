import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .utils import output_size_from_model_params, drop_last_layers
from . import CNN_pre, FFNN_pre



class EmbraceNet(nn.Module):
    
    def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):
        """
        Initialize an EmbraceNet module.
        Args:
          device: A "torch.device()" object to allocate internal parameters of the EmbraceNet module.
          input_size_list: A list of input sizes.
          embracement_size: The length of the output of the embracement layer ("c" in the paper).
          bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
        """
        super(EmbraceNet, self).__init__()

        self.device = device
        self.input_size_list = input_size_list
        self.embracement_size = embracement_size
        self.bypass_docking = bypass_docking

        if (not bypass_docking):
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))

    def forward(self, input_list, availabilities=None, selection_probabilities=None):
        #batch_size??
        """
        Forward input data to the EmbraceNet module.
        Args:
          input_list: A list of input data. Each input data should have a size as in input_size_list.
          availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
          selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
        Returns:
          A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
        """

        # check input data
        assert len(input_list) == len(self.input_size_list)
        num_modalities = len(input_list)
        batch_size = input_list[0].shape[0]


        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                x = getattr(self, 'docking_%d' % (i))(input_data)
                x = nn.functional.relu(x)
                docking_output_list.append(x)


        # check availabilities
        if (availabilities is None):
            availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        else:
            availabilities = availabilities.float()

        # adjust selection probabilities
        if (selection_probabilities is None):
            selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)

        selection_probabilities = torch.mul(selection_probabilities.float().to(self.device), availabilities.float().to(self.device))

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)


        # stack docking outputs
        docking_output_stack = torch.stack(docking_output_list, dim=-1)  # [batch_size, embracement_size, num_modalities]


        # embrace
        modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
        modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

        embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
        embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

        return embracement_output



class EmbraceNetMultimodal(nn.Module):
    def __init__(self, 
                 trial,
                 cell_line,
                 task,
                 device,
                 in_features_FFNN,
                 n_classes=2,  
                 args=None,
                 embracenet_dropout=True):
        super(EmbraceNetMultimodal, self).__init__()
        
        
        # input parameters
        self.trial=trial
        self.cell_line=cell_line
        self.device = device
        self.n_classes = n_classes
        self.embracenet_dropout=embracenet_dropout
        self.args = args

        # 1) pre neural networks
        self.FFNN = FFNN_pre(self.trial, in_features_FFNN, device=self.device) 
        self.CNN = CNN_pre(self.trial, device=self.device) 
        
        self.FFNN_pre_output_size = self.FFNN.output_size
        self.CNN_pre_output_size = self.CNN.output_size

        
        # 2) embracenet
        embracement_size = self.trial.suggest_categorical("EMBRACENET_embracement_size", [512, 768, 1024]) #????
        
        self.embracenet = EmbraceNet(device=self.device, 
                                     input_size_list=[self.FFNN_pre_output_size, self.CNN_pre_output_size], 
                                     embracement_size=embracement_size) #par to tune
        
        in_features = embracement_size
        
        
        # 3) post embracement layers
        n_post_layers = self.trial.suggest_int("n_post_layers", 0, 2) 
        post_layers = []

        for i in range(n_post_layers):
            if i==0:
                out_features = self.trial.suggest_categorical("EMBRACENET_n_units_l{}".format(i), [32, 64, 128, 256, 512])
            elif i==1:
                out_features = self.trial.suggest_categorical("EMBRACENET_n_units_l{}".format(i), [16, 32, 64, 128, 256])
                
            post_layers.append(nn.Linear(in_features, out_features))
            post_layers.append(nn.ReLU())
            
            dropout = self.trial.suggest_categorical("EMBRACENET_dropout_l{}".format(i), [0.0, 0.2, 0.3, 0.5])
            post_layers.append(nn.Dropout(dropout))
                
            in_features = out_features
            
        post_layers.append(nn.Linear(in_features, self.n_classes))
        # n of layers to tune

        self.post = nn.Sequential(*post_layers)

        selection_probabilities = self.trial.suggest_float("selection_probabilities_FFNN", 0.0, 1.0)
        self.selection_probabilities = torch.tensor([selection_probabilities, 1.0-selection_probabilities])
  
    def forward(self, x, availabilities=None, selection_probabilities=None, is_training=False, embracenet_dropout=True):
    # availabilities: don't change since we have all the data
    # selection probabilities: weight the contribution according to the auprc
    
        x_FFNN, x_CNN = x
        x_FFNN = self.FFNN(x_FFNN)
        x_CNN = self.CNN(x_CNN)

        # drop left or right modality #non serve
      #  availabilities = None
       # if (self.args.model_drop_left or self.args.model_drop_central or self.args.model_drop_right):
        #    availabilities = torch.ones([x.shape[0], 3], device=self.device)
         #   if (self.args.model_drop_left):
          #      availabilities[:, 0] = 0
           # if (self.args.model_drop_central):
            #    availabilities[:, 1] = 0
           # if (self.args.model_drop_right):
            #    availabilities[:, 2] = 0

        if (is_training and embracenet_dropout):
            dropout_prob = torch.rand(1, device=self.device)[0]
            if (dropout_prob >= 0.5):
                target_modalities = torch.round(torch.rand([x_FFNN.shape[0]], device=self.device)).to(torch.int64)
                availabilities = nn.functional.one_hot(target_modalities, num_classes=2).float()

        self.selection_probabilities_ = self.selection_probabilities.repeat(x_FFNN.shape[0],1)

        # embrace
        embracenet = self.embracenet([x_FFNN, x_CNN], availabilities=availabilities, selection_probabilities=self.selection_probabilities_)

        # employ final layers
        output = self.post(embracenet)

        # output softmax
        return output

