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
from scipy.sparse import coo_matrix
import gc
import re
import os


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
            for model in models:
                    
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



def get_average_AUPRC_df(models=['FFNN','CNN','ConcatNetMultimodal','EmbraceNetMultimodal','EmbraceNetMultimodal_augm'],
    rounding=3):
    
    if isinstance(models,str):
        models = [models]
    
    with open ('results_dict.pickle', 'rb') as fin:
        results_dict = pickle.load(fin)
        results_dict = defaultdict(lambda: defaultdict(dict), results_dict)
    
    cells_df_dict = defaultdict(lambda: defaultdict(list))

    for cell in CELL_LINES:
        df = pd.DataFrame(columns=TASKS, index=models)

        for task in TASKS:
            for model in models:
                try:
                    df.loc[model][task] = np.round(results_dict[cell][task][model]['average_CV_AUPRC'],rounding)
                except:
                    df.loc[model][task] = np.nan
                
        cells_df_dict[cell] = df
    
    return cells_df_dict



def get_standard_dev_df(models=['FFNN','CNN','ConcatNetMultimodal','EmbraceNetMultimodal','EmbraceNetMultimodal_augm'],
    rounding=3):
    
    if isinstance(models,str):
        models = [models]
    
    with open ('results_dict.pickle', 'rb') as fin:
        results_dict = pickle.load(fin)
        results_dict = defaultdict(lambda: defaultdict(dict), results_dict)
    
    cells_df_dict = defaultdict(lambda: defaultdict(list))

    for cell in CELL_LINES:
        df = pd.DataFrame(columns=TASKS, index=models)

        for task in TASKS:
            for model in models:
                l=[]

                for i in range(1,4):
                    l.append(results_dict[cell][task][model][f'iteration_n_{i}']['AUPRC_test'][-1]
                        )

                df.loc[model][task] = np.round(np.std(l), rounding)
                
        cells_df_dict[cell] = df
    
    return cells_df_dict


def dd():
    return defaultdict(dict)


class Compare_Models_Result():
    
    def __init__(self):
        self.models_dict = {'EmbraceNetMultimodal': EmbraceNetMultimodal_NoTrain,
                   'EmbraceNetMultimodal_augmentation': EmbraceNetMultimodal_NoTrain,
                  'ConcatNetMultimodal': ConcatNetMultimodal_NoTrain,
                  'FFNN': FFNN_NoTrain,
                  'CNN': CNN_NoTrain}
        self.pval_dict = defaultdict(dd)

        warnings.filterwarnings("ignore")
    
    
    def get_model_predictions(self, cell_line, task, model, n_iteration, device):

        model_ = self.models_dict[model]
        if model == 'CNN':
            model_ = model_(cell_line, task, n_iteration, device)
        else:
            if model.endswith('augmentation'):
                model_ = model_(cell_line, task, n_iteration, self.X_1.loc[0].shape[1], device=device, augmentation=True)
            else:
                model_ = model_(cell_line, task, n_iteration, self.X_1.loc[0].shape[1], device=device)

        state_dict = torch.load(f'{cell_line}_{model}_{task}_{n_iteration}_test_.pt', map_location=device)
                    
        model_.load_state_dict(state_dict['model_state_dict'])
        model_.double().to(device)

        for p in model_.parameters():
            p.require_grads = False

        model_.eval()

        with torch.no_grad():
            if model.startswith(UNIMODAL_NETWORKS_NOSEQ):
                output = torch.tensor([ model_( self.X_1.loc[i] )[1] for i in range(self.X_1.shape[0]) ])
            elif model.startswith(UNIMODAL_NETWORKS_SEQ):
                output = torch.tensor([ model_( self.X_2.loc[i] )[1] for i in range(self.X_2.shape[0]) ])
            elif model.startswith(MULTIMODAL_NETWORKS):
                output = torch.tensor([ model_(
                                            [self.X_1.loc[i],
                                             self.X_2.loc[i]]
                                    )[1] for i in range(self.X_1.shape[0]) ])

        return output


    def print_model_difference(self, p_val=0.05):

        self.counter_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
    
        for task in self.pval_dict.keys():
            for cell_line in self.pval_dict[task].keys():
                for fold in self.pval_dict[task][cell_line].keys():
                    for b_model in self.pval_dict[task][cell_line][fold].keys():
                        for c_model in self.pval_dict[task][cell_line][fold][b_model].keys():

                            if self.pval_dict[task][cell_line][fold][b_model][c_model] >=p_val:
                                self.counter_dict[task][cell_line][b_model][c_model]+=0
                            else:
                                self.counter_dict[task][cell_line][b_model][c_model]+=1
                
        
        for task in self.counter_dict.keys():
            print(f'\n\n================ TASK: {task} ================')
            for cell_line in self.counter_dict[task].keys():
                print(f'\n\n{cell_line}')
                for b_model in self.counter_dict[task][cell_line].keys():
                    print(f'\n\nBASE MODEL: {b_model}\n')
                    for c_model in self.counter_dict[task][cell_line][b_model].keys():

                        if self.counter_dict[task][cell_line][b_model][c_model] >=2:
                            print(f'{c_model} ===> different: True')
                        else:
                            print(f'{c_model} ===> different: False')



    def __call__(self, device, base_model='EmbraceNetMultimodal', 
                 comparison_models=['FFNN','CNN','ConcatNetMultimodal'],
                 augmentation_base_model=True, n_folds=3, cell_lines=CELL_LINES, tasks=TASKS,
                 pval_dict=None):

        if pval_dict:
            self.pval_dict = pval_dict

        else:
            if isinstance(base_model, str):
                base_model = [base_model]
            if isinstance(comparison_models, str):
                comparison_models = [comparison_models]
            if isinstance(tasks, str):
                tasks = [tasks]
            if isinstance(cell_lines, str):
                cell_lines = [cell_lines]


            MODELS = comparison_models + base_model
            if augmentation_base_model:
                MODELS += [f'{base_model[0]}_augmentation']
                base_model += [f'{base_model[0]}_augmentation']

            for task in tqdm(tasks, desc='Iter tasks'):

                if os.path.exists(f'pval_results_dict_{task}.pickle'):
                    with open (f'pval_results_dict_{task}.pickle', 'rb') as fin:
                        self.pval_dict = pickle.load(fin)
                        self.pval_dict = defaultdict(lambda: defaultdict(dict), self.pval_dict)

                pipe_data_load = Build_DataLoader_Pipeline(path_name=f'{task}.pickle')
                data_class = pipe_data_load.data_class

                for cell_line in tqdm(cell_lines, desc='Iter cell lines'):
                    _, self.X_1, _ = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=False)
                    _, self.X_2, _ = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=True)

                    self.X_1.reset_index(drop=True, inplace=True)
                    self.X_2.reset_index(drop=True, inplace=True)

                    self.X_1 = self.X_1.apply(lambda x: torch.reshape(torch.tensor(x), (1,-1)), axis=1) 
                    self.X_2 = self.X_2.apply(lambda x: torch.reshape(torch.tensor(process_sequence(x)), (1,4,256)) )


                    for i in tqdm(range(n_folds), desc='Iter folds'):
                        i+=1
                        self.pval_dict[task][cell_line][str(i)] = defaultdict(dd)
                        
                        for b_model in tqdm(base_model, desc='iter models'):
                            base_pred = self.get_model_predictions(cell_line, task, b_model, i, device)

                            for c_model in MODELS:
                                if c_model == b_model:
                                    continue
                                else:    
                                    c_pred = self.get_model_predictions(cell_line, task, c_model, i, device)
                                    pval = wilcoxon (base_pred, c_pred)[1]

                                    del c_pred
                                    self.pval_dict[task][cell_line][str(i)][b_model][c_model] = pval

                            del base_pred

                    del self.X_1
                    del self.X_2 
                
                    with open (f'pval_results_dict_{task}.pickle', 'wb') as fout:
                        pickle.dump(OrderedDict(self.pval_dict), fout)

                del pipe_data_load 
                #gc.collect()

        self.print_model_difference()

        return self.pval_dict



def parse_as_dict(X):
    X = re.split(': |\n',X)
    keys=[]
    vals=[]
    for i,x in enumerate(X):
        if i%2==0:
            keys.append(x.lstrip())
        else:
            try:
                vals.append(float(x))
            except:
                vals.append(x)
                
    d=defaultdict(float)
    
    for k,v in zip(keys,vals):
        d[k]=v
        
    return OrderedDict(d)



def parse_output_for_params_dict(output, cell_line, task, model, device, verbose=False, augmentation=False):

    params = []
    for match in re.finditer('Params:\s+', output):
        start = match.end()
        end = re.search('\n\n',output[start:]).start() + start
        params.append(output[start:end])

    for n in range(3):
        i=n+1
        if augmentation:
            NN_dict = torch.load(f'{cell_line}_{model.__name__}_augmentation_{task}_{i}_test_.pt', map_location=torch.device(device))
        else:
            NN_dict = torch.load(f'{cell_line}_{model.__name__}_{task}_{i}_test_.pt', map_location=torch.device(device))

        NN_dict['model_params']=parse_as_dict(params[n])
        if verbose:
            print(i)
            display(NN_dict['model_params'])
        
        if augmentation:
            torch.save(NN_dict, f'{cell_line}_{model.__name__}_augmentation_{task}_{i}_test_.pt')
        else:
            torch.save(NN_dict, f'{cell_line}_{model.__name__}_{task}_{i}_test_.pt')


def compare_model_overall_performance(base_model = ['EmbraceNetMultimodal','EmbraceNetMultimodal_augm'],
    compare_model = ['FFNN','CNN','ConcatNetMultimodal']):


    with open ('results_dict.pickle', 'rb') as fin:
        results_dict = pickle.load(fin)
        results_dict = defaultdict(lambda: defaultdict(dict), results_dict)


    BASE_MODEL = base_model
    COMPARE_MODEL = compare_model

    df_2sided = pd.DataFrame(columns=BASE_MODEL, index=COMPARE_MODEL+BASE_MODEL)
    df_1sided_greater = pd.DataFrame(columns=BASE_MODEL, index=COMPARE_MODEL+BASE_MODEL)
    df_1sided_less = pd.DataFrame(columns=BASE_MODEL, index=COMPARE_MODEL+BASE_MODEL)

    for b_model in BASE_MODEL:
        b_model_score = []
        
        for cell in CELL_LINES:
            for task in TASKS:
                b_model_score.append( results_dict[cell][task][b_model]['final_test_AUPRC_scores'] )
        
        for c_model in COMPARE_MODEL+BASE_MODEL:
            if c_model == b_model:
                continue
                
            c_model_score = []

            for cell in CELL_LINES:
                for task in TASKS:
                    c_model_score.append( results_dict[cell][task][c_model]['final_test_AUPRC_scores'] )
                

            df_2sided.loc[c_model][b_model] = np.round(
                    wilcoxon(
                        list(itertools.chain(*b_model_score)),
                        list(itertools.chain(*c_model_score))
                    )[1], 3)
                
            
            df_1sided_greater.loc[c_model][b_model] = np.round(
                wilcoxon(
                    list(itertools.chain(*b_model_score)),
                    list(itertools.chain(*c_model_score)),
                    alternative='greater'
                )[1], 3)
            
            # c_model median greater than b_model median
            
            df_1sided_less.loc[c_model][b_model] = np.round(
                wilcoxon(
                    list(itertools.chain(*b_model_score)),
                    list(itertools.chain(*c_model_score)),
                    alternative='less'
                )[1], 3)
            
            # c_model median lower than b_model median

    return df_2sided, df_1sided_greater, df_1sided_less
