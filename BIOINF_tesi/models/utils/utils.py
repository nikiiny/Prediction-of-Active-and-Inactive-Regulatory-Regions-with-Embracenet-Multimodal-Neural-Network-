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
 #   for key,value in model_param.copy().items():
  #    if re.findall('last', key):
   #     del model_param[str(key)]

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
    
    pos_inv = 1/pos if 1/pos !=0 else 0
    neg_inv = 1/neg if 1/neg !=0 else 0
    
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
    
    pos_inv = 1/pos if 1/pos !=0 else 0
    neg_inv = 1/neg if 1/neg !=0 else 0
    
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


def output_size_from_model_params(model_params):
    
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
    
    AUPRC_FFNN = results_dict[cell_line][task]['FFNN']['average_CV_AUPRC']
    AUPRC_CNN = results_dict[cell_line][task]['CNN']['average_CV_AUPRC']
    prob = torch.tensor([AUPRC_FFNN,AUPRC_CNN])
    prob = prob.repeat(batch_size,1)
    
    return prob



def drop_last_layers(model_state_dict, network_type):
    
    if network_type=='FFNN':
        keys = list(model_state_dict.keys())[-2:]
        for k in keys:
            del model_state_dict[k]
    
    elif network_type=='CNN':
        for k in model_state_dict.copy().keys():
            if k.startswith('last'):
                del model_state_dict[k]
                
    return model_state_dict




def select_augmented_models(results_dict, verbose=False, model_name='FFNN', augm_1='double', augm_2='smote'):

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



import seaborn as sns
import itertools


import seaborn as sns
import itertools


def plot_scores(cells, models=['FFNN','CNN'], k=3):

    TASKS=[]
    AUPRC=np.empty([0])
    MODEL=[]
    TEST_TRAIN=[]
    CELLS=[]
    
    if isinstance(cells, str):
        cells=[cells]
        
    for cell in cells:
        for task in results_dict[cell].keys():    
            for model in results_dict[cell][task].keys():
                if model in models:
                    
                    AUPRC=np.append(AUPRC, results_dict[cell][task][model]['final_train_AUPRC_scores'])
                    AUPRC=np.append(AUPRC, results_dict[cell][task][model]['final_test_AUPRC_scores'])
                    TEST_TRAIN.append(['train']*k), TEST_TRAIN.append(['test']*k) 
                    MODEL.append([model]*k*2)
                    TASKS.append([task]*k*2)
                    CELLS.append([cell]*k*2)
                    
                       # sns.barplot(y=y)
    MODEL=list(itertools.chain(*MODEL))
    TEST_TRAIN=list(itertools.chain(*TEST_TRAIN))
    TASKS=list(itertools.chain(*TASKS))
    CELLS=list(itertools.chain(*CELLS))
    data = {'AUPRC':AUPRC, 'model':MODEL, 'test_train':TEST_TRAIN, 'tasks':TASKS, 'cell':CELLS}
    p = pd.DataFrame.from_dict(data)
    
    colors=['#80d4ff','#ff3385']
    colors2=['#ff80d5','#aaff00']
    palette=sns.color_palette(colors2)

    sns.set_theme(style="whitegrid", font_scale=1.3)
    plot = sns.catplot(y='tasks', x='AUPRC',hue='test_train',row='model', data=p, kind="bar", orient='h',
           height=5, aspect=2, palette=colors2 , legend_out=False, col='cell')  #Set2 #cool
    plot.set_ylabels('', fontsize=15)
    plot.set(xlim=(0,1))
    plot.set_titles('{row_name}' ' | ' '{col_name}')