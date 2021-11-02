import pandas as pd
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from BIOINF_tesi.data_pipe import Build_DataLoader_Pipeline

TASKS = ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 
                        'active_E_vs_active_P', 'inactive_E_vs_inactive_P',
                        'active_EP_vs_inactive_rest']

CELL_LINES = ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']


def plot_label_ratio(task, title=None):

    if task not in TASKS:
            raise ValueError(
                f"Argument 'task' has an incorrect value: use one among {TASKS} ")

    pipe_data_load = Build_DataLoader_Pipeline(path_name=f'{task}.pickle')
    labels_dict = pipe_data_load.data_class.labels_dict
    fig, axes = plt.subplots(2,4, figsize=[20,10])

    row=0
    col=0

    fig.suptitle(title, fontsize=20)

        
    for cell,ax in zip(CELL_LINES, axes.flatten()):
        x = np.unique(labels_dict[cell], return_counts=True)[1]
        ax.pie(x=x, autopct="%.1f%%", explode=[0.03]*2,  labels=['0','1'], pctdistance=0.5, 
               colors=['#A9A9A9','#32CD32'],textprops={'fontsize': 15})
        ax.set_title(str(cell), fontsize=16)
        ax

        col+=1
        if col==4:
            row+=1
            col=0
            
    fig.delaxes(axes[row][col])


def get_imbalance_ratio_df():
    
    df = pd.DataFrame(columns=TASKS, index=CELL_LINES)
    
    for task in TASKS:
        pipe_data_load = Build_DataLoader_Pipeline(path_name=f'{task}.pickle')
        labels_dict = pipe_data_load.data_class.labels_dict
        
        for cell in CELL_LINES:
            pos = len(labels_dict[cell][labels_dict[cell]==1])
            neg = len(labels_dict[cell][labels_dict[cell]==0])
            
            df.loc[cell][task] = np.round(neg/pos,3)
    
    return df



def get_baseline_df():
    
    df = pd.DataFrame(columns=TASKS, index=CELL_LINES)
    
    for task in TASKS:
        pipe_data_load = Build_DataLoader_Pipeline(path_name=f'{task}.pickle')
        labels_dict = pipe_data_load.data_class.labels_dict
        
        for cell in CELL_LINES:
            pos = len(labels_dict[cell][labels_dict[cell]==1])
            tot = len(labels_dict[cell])
            
            baseline = np.round(pos/tot,3)
            df.loc[cell][task] = baseline if baseline >=0.1 else 0.1
    
    return df



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
           height=4, aspect=2.5, palette=PALETTE[palette] , legend_out=False, col='cell', ci='sd')  
    plot.set_ylabels('', fontsize=15)
    plot.set(xlim=(0,1))
    plot.set_titles('{col_name}' ' | ' '{row_name}')

    axes = plot.axes.flatten()
    for i,ax in enumerate(axes):
        ax.axvline(baseline[i], color='red',linewidth=3, ls='--')



def print_content_results_dict(models=['FFNN','CNN','EmbraceNetMultimodal','ConcatNetMultimodal', 'EmbraceNetMultimodal_augm']):
    with open ('results_dict.pickle', 'rb') as fin:
        results_dict = pickle.load(fin)
        results_dict = defaultdict(lambda: defaultdict(dict), results_dict)
    
    if isinstance(models, str):
        models=[models]

    for cell in results_dict.keys():
        print(cell)
        for task in results_dict[cell].keys():
            print(f'\n{task}')
            for key in results_dict[cell][task].keys():
                if key in models:
                    print(key)
        print('\n')



def get_average_AUPRC_df(models=['FFNN','CNN','EmbraceNetMultimodal','ConcatNetMultimodal']):
    
    if isinstance(models,str):
        models = [models]
    
    with open ('results_dict.pickle', 'rb') as fin:
        results_dict = pickle.load(fin)
        results_dict = defaultdict(lambda: defaultdict(dict), results_dict)
    
    models_df_dict = defaultdict(lambda: defaultdict(list))
    for model in models:
        df = pd.DataFrame(columns=TASKS, index=CELL_LINES)
    
        for task in TASKS:
            for cell in CELL_LINES:
                try:
                    df.loc[cell][task] = results_dict[cell][task][model]['average_CV_AUPRC']
                except:
                    df.loc[cell][task] = np.nan
                
        models_df_dict[model] = df
    
    return models_df_dict
