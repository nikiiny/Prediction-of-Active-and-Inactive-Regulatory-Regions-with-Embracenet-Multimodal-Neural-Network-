import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pylab as plt

import os, shutil
import pickle
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import ranksums



class EarlyStopping():
    """Early stops the training if validation score doesn't improve after a given patience.
    Modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    
    Parameters:
    ------------------
        patience (int): How long to wait after last time validation score improved.
            Default: 4
        verbose (bool): If True, prints a message for each validation score improvement. 
            Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
        trace_func (function): trace print function.
            Default: print 
                            
    Attributes:
    ------------------
        early_stop (bool): True if the validation score doesn't improveand the training should
            be stopped, False else.
        """
    
    def __init__(self, patience=4, verbose=False, delta=0, trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
        # if the new score is worse than the previous score, add 1 to the counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # if the number of non-improving epochs is greater than patience, 
            #set to True early_stop attribute 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



def accuracy(output, target):
    """Computes accuracy for output of a neural network."""
    pred = torch.argmax(output, dim=1)
    # return the category with the highest probability
    return (pred == target).float().mean()
    # return true if the predicted category is equal to the true one, false otherwise.
    #we transform them in float, 1 if True, 0 is False, then we return the mean of the vector


def AUPRC(output, target):
    """Computes AUPRC for output of a neural network."""
    pred = torch.argmax(output, dim=1).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    res = average_precision_score(target, pred) 
    
    return res if not np.isnan(res) else 0


def F1_precision_recall(output, target):
    """Computes F1, precision and recall for output of a neural network."""
    pred = torch.argmax(output, dim=1).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    return np.array(precision_recall_fscore_support(target, pred, average='macro', zero_division=0)[:3])



def get_loss_weights_from_dataloader(dataloader):
    """
    Returns normalized weights of positive and negative class according
    to Inverse Number of Samples (INS) from a DataLoader object.

    Parameters:
    ------------------
        dataloader: Pytorch DataLoader.
    """

    pos=0
    tot=0
    for i,j in dataloader:
        pos+=j.sum()
        tot+=len(j)
    neg=tot-pos
    
    pos_inv = 1/pos if pos !=0 else 0
    neg_inv = 1/neg if neg !=0 else 0
    
    return pos_inv/(neg_inv+pos_inv), neg_inv/(neg_inv+pos_inv)


def get_loss_weights_from_labels(label):
    """
    Returns normalized weights of positive and negative class according
    to Inverse Number of Samples (INS) from Series of labels.

    Parameters:
    ------------------
        labels (pd.series): binary labels.
    """

    if isinstance(label, pd.DataFrame):
        label=pd.Series(label.values)
    
    pos = len(label[label==1])
    neg = len(label[label==0])
    
    pos_inv = 1/pos if pos !=0 else 0
    neg_inv = 1/neg if neg !=0 else 0
    
    return pos_inv/(neg_inv+pos_inv), neg_inv/(neg_inv+pos_inv)


def size_out_convolution(input_size, kernel, padding, stride):
    """Calculates and returns the size of input after a 1D convolution.

    Parameters:
    ------------------
        input_size (int): input size of the convolution.
        kernel (int): kernel of the convolution.
        padding (int): padding of hte convolution.
        stride (int): stride of the convolution.
    """
    return int(( (input_size + 2*padding - kernel)/stride )+1)

def weight_reset(x):
    """Resets the weights of neural networks.

    Parameters:
    ------------------
        x (torch.nn): layers of the network.
    """
    if isinstance(x, nn.Conv1d) or isinstance(x, nn.Linear) or isinstance(x, nn.LSTM):
        x.reset_parameters()

def get_input_size(data_loader):
    """Returns the size of the input from a DataLoader object

    Parameters:
    ------------------
        dataloader: Pytorch DataLoader.
    """
    for d,l in data_loader:
        input_size = d.shape[1]
        break
    return input_size


def output_size_from_model_params(model_params):
    """Returns the size of the fully connected layer after convolutions
    from the model parameters
    
    Parameters:
    ------------------
        model_params (dict): dictionary containing the parameters of the model.
    """
    n_layers = model_params['n_layers']
    input_size=256

    for i in range(n_layers):
        kernel_size = model_params[f'kernel_size_l{i}']
        padding = int((kernel_size-1)/2)

        maxpool_kernel_size = 10
        maxpool_stride = 2

        output_size = size_out_convolution(input_size, kernel_size , padding, 1)
        output_size = size_out_convolution(output_size, maxpool_kernel_size, 0, maxpool_stride)
        input_size=output_size

        out_channels=model_params[f'out_channels_l{i}']
        
    return output_size*out_channels



def selection_probabilities(results_dict, cell_line, task, batch_size):
    """Returns the values of the selection_probabilities for the EmbraceNet
    based on the average AUPRC of the single networks

    Parameters:
    ------------------
        results_dict (dict): dictionary containing scores of the models.
        cell_line (str): name of the cell line.
        task (str): name of the task.
        batch_size (int): size of the batch.

    Returns:
    ------------------
        tensor with shape [n.modalities, batch size] containing probabilities for modalities.
    """
    AUPRC_FFNN = results_dict[cell_line][task]['FFNN']['average_CV_AUPRC']
    AUPRC_CNN = results_dict[cell_line][task]['CNN']['average_CV_AUPRC']
    prob = torch.tensor([AUPRC_FFNN,AUPRC_CNN])
    prob = prob.repeat(batch_size,1)
    
    return prob



def drop_last_layers(model_state_dict, network_type):
    """Drop last layers for loading pre-trained feed-forward and CNN in a 
    multimodal network

    Parameters:
    ------------------
        model_state_dict (dict): Pytorch state_dict.
        network_type (str): name of the network.
    """
    if network_type=='FFNN':
        keys = list(model_state_dict.keys())[-2:]
        for k in keys:
            del model_state_dict[k]
    
    elif network_type=='CNN':
        for k in model_state_dict.copy().keys():
            if k.startswith('last'):
                del model_state_dict[k]
                
    return model_state_dict




def select_augmented_models(results_dict, verbose=False, model_name='FFNN', augm_1='smote', augm_2='double'):
    """Selects the best augmented models by comparing their scores through the wilcoxon test. If the difference
    is not significant, keeps by default augm_1.
    Saves the best performing model in the results_dict object and copies and saves the best model.

    Parameters:
    ------------------
        results_dict (dict): dictionary containing scores of the models. 
        verbose (bool): whether to print or not info about the statistical test.
            Default: False
        model_name (str): name of the model.
            Default: 'FFNN'
        augm_1 (str): type of augmentation n.1
            Default: 'double'
        augm_2 (str): type of augmentation n.2
            Default: 'smote'
    """

    for cell in results_dict.keys():
        for task in results_dict[cell].keys():
            if set({f'{model_name}_{augm_1}',f'{model_name}_{augm_2}'}).issubset({*results_dict[cell][task].keys()}):
                pval = ranksums(results_dict[cell][task][f'{model_name}_{augm_1}']['final_test_AUPRC_scores'], results_dict[cell][task][f'{model_name}_{augm_2}']['final_test_AUPRC_scores'])[1]
                if verbose:
                    print(f'\n{cell}')
                    print(task)
                    print(f'pvalue: {pval}')

                if pval<0.3 and results_dict[cell][task][f'{model_name}_{augm_2}']['average_CV_AUPRC'] >= results_dict[cell][task][f'{model_name}_{augm_1}']['average_CV_AUPRC']:
                        results_dict[cell][task][model_name] = results_dict[cell][task][f'{model_name}_{augm_2}'].copy()
                        results_dict[cell][task]['best_augmentation']=augm_2
                        shutil.copy(f'models/{cell}_{task}_{model_name}_{augm_2}_TEST.pt', 
                                      f'models/{cell}_{task}_{model_name}_TEST.pt')
                        if verbose:
                            print(f'Best augmentation method: {augm_2}')

                else:
                    results_dict[cell][task][model_name] = results_dict[cell][task][f'{model_name}_{augm_1}'].copy()
                    results_dict[cell][task]['best_augmentation']=augm_2 #SISTEMA IN CV
                    shutil.copy(f'models/{cell}_{task}_{model_name}_{augm_1}_TEST.pt', 
                                          f'models/{cell}_{task}_{model_name}_TEST.pt')
                    if verbose:
                        print(f'Best augmentation method: {augm_1}')

    return results_dict



def plot_scores(cells, models=['FFNN','CNN'], k=3, palette=1):
    """Plot training and testing scores of the models by cell line and task.
    
    Parameters:
    ------------------
        cells (str, list): cell line.
        models (str, list): name of the model to plot.
            Default: ['FFNN','CNN']
        k (int): number of k-folds.
            Default: 3
        palette (int): palette of color.
            Default: 1
    """

    TASKS=[]
    AUPRC=np.empty([0])
    MODEL=[]
    TEST_TRAIN=[]
    CELLS=[]

    baseline=[]
    
    if isinstance(cells, str):
        cells=[cells]
    
    with open ('results_dict.pickle', 'rb') as fin:
        results_dict = pickle.load(fin)
        results_dict = defaultdict(lambda: defaultdict(dict), results_dict)

    # create suitable dataframe for plotting data from dict. 
    for cell in cells:
        for task in results_dict[cell].keys():    
            # store baseline AUPRC
            baseline.append(results_dict[cell][task]['baseline_AUPRC'])
            for model in results_dict[cell][task].keys():
                if model in models:
                    
                    AUPRC=np.append(AUPRC, results_dict[cell][task][model]['final_train_AUPRC_scores'])
                    AUPRC=np.append(AUPRC, results_dict[cell][task][model]['final_test_AUPRC_scores'])
                    TEST_TRAIN.append(['train']*k), TEST_TRAIN.append(['test']*k) 
                    MODEL.append([model]*k*2)
                    TASKS.append([task]*k*2)
                    CELLS.append([cell]*k*2)
                    
    MODEL=list(itertools.chain(*MODEL))
    TEST_TRAIN=list(itertools.chain(*TEST_TRAIN))
    TASKS=list(itertools.chain(*TASKS))
    CELLS=list(itertools.chain(*CELLS))
    data = {'AUPRC':AUPRC, 'model':MODEL, 'test_train':TEST_TRAIN, 'tasks':TASKS, 'cell':CELLS}
    p = pd.DataFrame.from_dict(data)
    
    PALETTE = [
                sns.color_palette(['#80d4ff','#ff3385']),
                sns.color_palette(['#ff80d5','#aaff00']),
                'Set2'
            ]

    sns.set_theme(style="whitegrid", font_scale=1.3)
    plot = sns.catplot(y='model', x='AUPRC',hue='test_train',row='tasks', data=p, kind="bar", orient='h',
           height=4, aspect=2.5, palette=PALETTE[palette] , legend_out=False, col='cell')  
    plot.set_ylabels('', fontsize=15)
    plot.set(xlim=(0,1))
    plot.set_titles('{col_name}' ' | ' '{row_name}')
    

    axes = plot.axes.flatten()
    for i,ax in enumerate(axes):
        ax.axvline(baseline[i], color='red',linewidth=3, ls='--')

