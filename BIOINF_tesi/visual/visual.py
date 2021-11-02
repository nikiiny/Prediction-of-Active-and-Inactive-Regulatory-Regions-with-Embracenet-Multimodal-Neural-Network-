import pandas as pd
import numpy as np
import torch
import itertools
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import wilcoxon 
from tqdm.auto import tqdm
from BIOINF_tesi.models import EmbraceNetMultimodal_NoTrain, ConcatNetMultimodal_NoTrain, FFNN_NoTrain, CNN_NoTrain
from BIOINF_tesi.data_pipe import Build_DataLoader_Pipeline
from BIOINF_tesi.data_pipe.utils import process_sequence
import warnings

TASKS = ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 
                        'active_E_vs_active_P', 'inactive_E_vs_inactive_P',
                        'active_EP_vs_inactive_rest']

CELL_LINES = ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']

UNIMODAL_NETWORKS_SEQ=('CNN')
UNIMODAL_NETWORKS_NOSEQ=('FFNN')
MULTIMODAL_NETWORKS=('EmbraceNetMultimodal', 'ConcatNetMultimodal')


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


class Compare_Models_Result():
    
    def __init__(self):
        self.models_dict = {'EmbraceNetMultimodal': EmbraceNetMultimodal_NoTrain,
                   'EmbraceNetMultimodal_augmentation': EmbraceNetMultimodal_NoTrain,
                  'ConcatNetMultimodal': ConcatNetMultimodal_NoTrain,
                  'FFNN': FFNN_NoTrain,
                  'CNN': CNN_NoTrain}
        self.prediction_dict = defaultdict(lambda: defaultdict(dict))
        self.pval_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        warnings.filterwarnings("ignore")
    
    
    def __call__(self, device, base_model='EmbraceNetMultimodal', 
                 comparison_models=['FFNN','CNN','ConcatNetMultimodal'],
                 augmentation_base_model=True):
    
        if isinstance(base_model, str):
            base_model = [base_model]
        if isinstance(comparison_models, str):
            comparison_models = [comparison_models]

        for task in tqdm(TASKS, desc='Iter tasks'):
            pipe_data_load = Build_DataLoader_Pipeline(path_name=f'{task}.pickle')
            data_class = pipe_data_load.data_class

            for cell_line in tqdm(CELL_LINES, desc='Iter cell lines'):
                _, X_1, _ = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=False)
                _, X_2, _ = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=True)

                X_1 = X_1.apply(lambda x: torch.reshape(torch.tensor(x), (1,-1)), axis=1)
                X_2 = X_2.apply(lambda x: torch.reshape(torch.tensor(process_sequence(x)), (1,4,256)) )

                MODELS = comparison_models + base_model
                if augmentation_base_model:
                    MODELS += [f'{base_model[0]}_augmentation']
                    base_model += [f'{base_model[0]}_augmentation']

                for model in tqdm(MODELS, desc='Iter models'):
                    model_ = self.models_dict[model]
                    if model == 'CNN':
                        model_ = model_(cell_line, task, device)
                    else:
                        if model.endswith('augmentation'):
                            model_ = model_(cell_line, task, X_1.loc[0].shape[1], device=device, augmentation=augmentation_base_model)
                        else:
                            model_ = model_(cell_line, task, X_1.loc[0].shape[1], device=device)

                    state_dict = torch.load(f'models/{cell_line}_{task}_{model}_TEST.pt', map_location=device)
                    
                    model_.load_state_dict(state_dict['model_state_dict'])
                    model_.double().to(device)

                    for p in model_.parameters():
                        p.require_grads = False

                    if model.startswith(UNIMODAL_NETWORKS_NOSEQ):
                        output = torch.tensor([ model_( X_1.loc[i] )[1] for i in range(len(X_1)) ])
                    elif model.startswith(UNIMODAL_NETWORKS_SEQ):
                        output = torch.tensor([ model_( X_2.loc[i] )[1] for i in range(len(X_2)) ])
                    elif model.startswith(MULTIMODAL_NETWORKS):
                        output = torch.tensor([ model_(
                                                    [X_1.loc[i],
                                                     X_2.loc[i]]
                                            )[1] for i in range(len(X_1)) ])

                    output = output.detach().numpy()
                    self.prediction_dict[task][cell_line][model] = output
                    
                    
                for b_model in base_model:
                    for c_model in comparison_models:
                        pval = wilcoxon( self.prediction_dict[task][cell_line][b_model],
                                        self.prediction_dict[task][cell_line][c_model] )[1]
                        self.pval_dict[task][cell_line][b_model][c_model] = pval
                        
                        
        return self.pval_dict
    
