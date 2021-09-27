import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import os
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



class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience.
    Modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    
    Parameters:
    ------------------
        patience (int): How long to wait after last time validation loss improved.
            Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement. 
            Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
        trace_func (function): trace print function.
            Default: print 
                            
    Attributes:
    ------------------
        early_stop (bool): True if the validation loss doesn't improveand the training should
            be stopped, False else.
        """
    
    def __init__(self, patience=5, verbose=False, delta=0, trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss
        
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



def load_model(model, path):
  """Load the stored weights of a pre-trained model into another
      model and set it to eval state.
    
    Parameters:
    ------------------
    model (torch.nn.Module): not trained neural network model.
    path (str): path of the stored weights of the pre-trained model. 
    """

  gdrive_path = '/content/gdrive/MyDrive/Thesis_BIOINF' ###
  basepath = 'models'
  basepath = gdrive_path + basepath ###

  path = os.path.join(basepath, path)
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint) 
  # set the model in testing modality
  model.eval() 



def save_best_model(model, path):
    """Saves only the weights of the common layers of a
    trained neural network. 
    
    Parameters:
    ------------------
    model (torch.nn.Module): trained neural network model.
    path (str): path where the weights of the trained model will be stored. 
    """
    
    model_param = model.state_dict()
    for key,value in model_param.copy().items():
      if re.findall('last', key):
        del model_param[str(key)]

    basepath = 'models'
    basepath = basepath 
    PATH = os.path.join(basepath, path)
    
    torch.save(model_param, PATH)




def plot_F1_scores(y_train, y_test, set_ylim=None):
    """Plots the trend of the training and test loss function of 
        a model.
    
    Parameters:
    ------------------
    y_train (list): list of training losses.
    y_test (list): list of test losses.
    set_ylim (tuple of int): range of y-axis.
        Default: None
    """
   
    epochs = range(len(y_train))
    X=pd.DataFrame({'epochs':epochs,'y_train':y_train,'y_test':y_test})
   
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(30,15)})

    f, ax = plt.subplots(1, 1)

    sns.lineplot(data=X, x="epochs", y="y_test", color='#dc143c',lw=2.5)
    sns.lineplot(data=X, x="epochs", y="y_train", color='#00bfff',lw=2.5)

    plt.legend(labels=['F1 test score', 'F1 train score'])
    plt.setp(ax.get_legend().get_texts(), fontsize=35)
    plt.setp(ax.get_legend().get_title(),fontsize=35)

    ax.set_ylabel('F1 score', fontsize=30)
    ax.set_xlabel('Epochs', fontsize=30)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_ylim(set_ylim)

    plt.show()



def plot_other_scores(AUPRC_prec_rec, set_ylim=None):
    """Plots the trend of the training and test loss function of 
        a model.
    
    Parameters:
    ------------------
    y_train (list): list of training losses.
    y_test (list): list of test losses.
    set_ylim (tuple of int): range of y-axis.
        Default: None
    """
   

    AUPRC, precision, recall = AUPRC_prec_rec
    epochs = range(len(AUPRC))
    X=pd.DataFrame({'epochs':epochs,'AUPRC':AUPRC,'Precision':precision,'Recall':recall})
   
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(30,15)})

    f, ax = plt.subplots(1, 1)

    sns.lineplot(data=X, x="epochs", y="AUPRC", color='#00ced1',lw=2.5)
    sns.lineplot(data=X, x="epochs", y="Precision", color='#ff1493',lw=2.5)
    sns.lineplot(data=X, x="epochs", y="Recall", color='#7b68ee',lw=2.5)

    plt.legend(labels=['AUPRC', 'Precision', 'Recall'])
    plt.setp(ax.get_legend().get_texts(), fontsize=35)
    plt.setp(ax.get_legend().get_title(),fontsize=35)

    ax.set_ylabel('Scores', fontsize=30)
    ax.set_xlabel('Epochs', fontsize=30)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_ylim(set_ylim)

    plt.show()


def accuracy(output, target):
  pred = torch.argmax(output, dim=1)
  # return the category with the highest probability
  return (pred == target).float().mean()
  # return true if the predicted category is equal to the true one, false otherwise.
  #we transform them in float, 1 if True, 0 is False, then we return the mean of the vector


def AUPRC(output, target):
    pred = torch.argmax(output, dim=1).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    res = average_precision_score(target, pred) 
    
    return res if not np.isnan(res) else 0


def F1_precision_recall(output, target):
    pred = torch.argmax(output, dim=1).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    return np.array(precision_recall_fscore_support(target, pred, average='macro', zero_division=0)[:3])



def get_loss_weights_from_dataloader(dataloader):
    """
    Returns normalized weights of positive and negative class according
    to Inverse Number of Samples (INS) from a DataLoader object.
    """
    pos=0
    tot=0
    for i,j in dataloader:
        pos+=j.sum()
        tot+=len(j)
    neg=tot-pos
    
    pos_inv = 1/pos
    neg_inv = 1/neg
    
    return pos_inv/(neg_inv+pos_inv), neg_inv/(neg_inv+pos_inv)


def get_loss_weights_from_labels(label):
    """
    Returns normalized weights of positive and negative class according
    to Inverse Number of Samples (INS) from Series of labels.
    """

    if isinstance(label, pd.DataFrame):
        label=pd.Series(label.values)
    
    pos = len(label[label==1])
    neg = len(label[label==0])
    
    pos_inv = 1/pos
    neg_inv = 1/neg
    
    return pos_inv/(neg_inv+pos_inv), neg_inv/(neg_inv+pos_inv)




def size_out_convolution(input_size, kernel, padding, stride):
    """Calculates and returns the size of input after a convolution"""
    return int(( (input_size + 2*padding - kernel)/stride )+1)

def weight_reset(x):
    if isinstance(x, nn.Conv1d) or isinstance(x, nn.Linear) or isinstance(x, nn.LSTM):
        x.reset_parameters()

def get_input_size(data_loader):
  for d,l in data_loader:
    input_size = d.shape[1]
    break
  return input_size

