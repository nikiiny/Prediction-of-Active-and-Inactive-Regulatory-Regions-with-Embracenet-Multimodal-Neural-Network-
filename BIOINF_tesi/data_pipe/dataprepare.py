import os 
from tqdm.notebook import tqdm
import re
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split, KFold
from collections import OrderedDict, defaultdict

from .utils import (MICE, kruskal_wallis_test, wilcoxon_test, spearman_corr, remove_correlated_features,
    data_augmentation)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

CELL_LINES = ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']

g_gpu = torch.Generator(device)


class Data_Prepare():
    """Applies robust scaler and MICE imputer to genomic features. 
    Checks for highly correlated features (Spearman correlation) and features not 
    correlated to the output (Point Biserial correlation or/and Logistic regression AUPRC).
    Splits dataset either for hyperparameters tuning or model testing.

    Parameters
    ---------------
    data_dict (dict): dictionary of cell lines (pd.DataFrame).
    labels_dict (dict): dictionary of cell lines labels (pd.Series).
    kruskal_pval_threshold: kruskal-wallis p-value threshold for correlation
        between X and y.
        Default: 0.05
    wilcoxon_pval_threshold: wilcoxon signed-rank p-value threshold for
        correlation between X and y.
        Default: 0.05
    spearman_corr_threshold: Spearman correlation threshold for correlation
        between different features.
        Default: 0.75
    """
    
    def __init__(self, 
                 data_dict, 
                 labels_dict, 
                 kruskal_pval_threshold = 0.05,
                 wilcoxon_pval_threshold = 0.05,
                 spearman_corr_threshold=0.75
                ):
        
        self.labels_dict = labels_dict.copy()
        self.index = data_dict['H1'][['chrom','chromStart','chromEnd','strand']].copy()
        self.data_dict = data_dict.copy()
        
        self.data_dict['fa'] = self.data_dict['fa']['chromosome']
        
        # drop observations info
        for key in self.data_dict.keys():
            if key != 'fa':
                self.data_dict[key] = self.data_dict[key].drop(['chrom','chromStart','chromEnd','strand'], axis=1)
                
        
        self.kruskal_pval_threshold = kruskal_pval_threshold
        self.wilcoxon_pval_threshold = wilcoxon_pval_threshold
        self.spearman_corr_threshold = spearman_corr_threshold
        
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()
        
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
        self.sequence = []
        
        self.to_drop = defaultdict(set)
               
    def scale_data_genfeatures(self):
        """Applies robust scaler to numeric genomic features by row/column?"""
        for key in self.data_dict.keys():
            if key != 'fa':
                self.data_dict[key] = pd.DataFrame(
                    self.minmax_scaler.fit_transform(self.robust_scaler.fit_transform(self.data_dict[key].values)),
                    index=self.data_dict[key].index, 
                    columns=self.data_dict[key].columns)
                
    
    def mice_imputation_genfeatures(self):
        """Applies MICE imputation to numeric genomic features."""
        for key in self.data_dict.keys():
            if key != 'fa':
                # if there are null values apply MICE, else pass
                try:
                    self.data_dict[key] = MICE(self.data_dict[key])
                except:
                    pass

    
    def transform(self):
        
        self.scale_data_genfeatures()
        self.mice_imputation_genfeatures()
        
    
    
    
    def correlation_with_label(self, type_test='kruskal_wallis_test', intersection=False, verbose=False):
        """Checks correlation of features with label and deletes uncorrelated columns.

        Parameters
        ----------------
        type_test (str or list of str): type of correlation test. Values are ['kruskal_wallis_test', 'wilcoxon_test'].
        intersection (bool): whether to remove the uncorrelated features selected by all the methods (intersection)
            or the uncorrelated features selected by at least one method (union).
            Default: False
        verbose (bool): returns info.
            Default: False
        """
        
        if isinstance(type_test, str):
            type_test = [type_test]
        if not set(type_test).issubset({'kruskal_wallis_test', 'wilcoxon_test'}):
            raise ValueError(
            "Argument 'type_test' has an incorrect value: use 'kruskal_wallis_test', 'wilcoxon_test'")
        
        # for the intersection we create a dictionary to store the uncorrelated columns
        #for every method, then we merge them.
        if intersection:
            for key in self.data_dict.keys():
                self.to_drop[key]= dict()
    
            
        if 'kruskal_wallis_test' in type_test:
            for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print(key)
                    cols_to_drop = kruskal_wallis_test(self.data_dict[key], self.labels_dict[key], self.kruskal_pval_threshold, verbose=verbose) 
                    # if we want intersection of uncorrelated features
                    if intersection:
                        #stores the features in a dictionary
                         self.to_drop[key]['kruskal_wallis_test'] = cols_to_drop
                    # if we want union of uncorrelated features
                    else:
                        self.to_drop[key] = self.to_drop[key].union(cols_to_drop)
        
        if 'wilcoxon_test' in type_test:
            for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print(key)
                    cols_to_drop = wilcoxon_test(self.data_dict[key], self.labels_dict[key], self.wilcoxon_pval_threshold, verbose=verbose) 
                    # if we want intersection of uncorrelated features
                    if intersection:
                        #stores the features in a dictionary
                        self.to_drop[key]['wilcoxon_test'] = cols_to_drop
                    # if we want union of uncorrelated features
                    else:
                        self.to_drop[key] = self.to_drop[key].union(cols_to_drop)
        
        
        # drop all the resulting incorrelated keys
        for key in self.data_dict.keys():
            if key != 'fa':
                # if intersection merge all the sets
                if intersection:
                    self.to_drop[key] = set.intersection(*self.to_drop[key].values())
                #drop columns
                if verbose:
                    print(f'\nColumns to drop for {key}: {self.to_drop[key]}')
                self.data_dict[key] = self.data_dict[key].drop(list(self.to_drop[key]), axis=1)
        
    
    
    
    def correlation_btw_features(self, type_test='wilcoxon_test',  verbose=False):
        """Runs Spearman correlation and remove less correlated feature from a pair
        of correlated features."""
        
        self.to_drop = dict()
        
        for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print('\n', key)
                    correlated_pairs = spearman_corr(self.data_dict[key], self.spearman_corr_threshold, verbose=verbose)
                    self.data_dict[key] =  remove_correlated_features(self.data_dict[key], self.labels_dict[key], 
                                                                      correlated_pairs, type_test=type_test, verbose=verbose)
                    
                    
    
    def split_data(self, cell_line, hyper_tuning, sequence, test_size, validation_size, random_state):
        """Splits data into training and test set for model testing. Splits further the training
        set and discard the test set if used for hyperparameters tuning.

        Parameters
        --------------
        cell_line (str): name of the cell line. Possible values are ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']
        hyper_tuning (bool): if True, creates a training and validation set
            and discards test set.
            Default: False
        sequence (bool): if the data passed are the genomic sequence.
        test_size (float): size of test set
        validation_size (float): size of validation set
        random_state (int): initial random seed for dataset split.
            Default: 123

        Returns
        ---------------
        Training set, Test set, training labels, test labels
        """
        
        if sequence:
            # if the task is active_E_vs_active_P or inactive_E_vs_inactive_P we need to select
            #the observations in fa corresponding to the labels index of the cell line
            if 'index_fa' in self.labels_dict:
                index_cell_line = self.labels_dict['index_fa'][cell_line]
                data_fa = self.data_dict['fa'].iloc[index_cell_line].reset_index(drop=True)
            else:
                data_fa = self.data_dict['fa']
                
            assert (data_fa.shape[0] ==  len(self.labels_dict[cell_line]))
                    
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_fa, 
                                                                                 self.labels_dict[cell_line],
                                                                                 test_size=test_size, 
                                                                                 random_state=random_state, shuffle=True) 

            if hyper_tuning:
                assert (self.X_train.shape[0] ==  len(self.y_train))
                    
                self.X_train, self.X_test,  self.y_train, self.y_test = train_test_split(self.X_train, 
                                                                                     self.y_train,
                                                                                     test_size=validation_size,
                                                                                     random_state=random_state+100, shuffle=True) 
                   
            
        else:
            assert (self.data_dict[cell_line].shape[0] ==  len(self.labels_dict[cell_line]))

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_dict[cell_line], 
                                                                                    self.labels_dict[cell_line],
                                                                                    test_size=test_size, 
                                                                                    random_state=random_state, shuffle=True) 
            
            if hyper_tuning: 
                assert (self.X_train.shape[0] ==  len(self.y_train))

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, 
                                                                                        self.y_train,
                                                                                        test_size=validation_size, 
                                                                                        random_state=random_state+100, shuffle=True) 
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        
    
    def return_index_data_for_cv(self,
                                cell_line,
                                sequence=False,
                                n_folds=3,
                                random_state=123):
        
        if cell_line not in CELL_LINES:
            raise ValueError(
            f"Argument 'cell_line' has an incorrect value: use one among {CELL_LINES}")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)


        if sequence:
            # if the task is active_E_vs_active_P or inactive_E_vs_inactive_P we need to select
            #the observations in fa corresponding to the labels index of the cell line
            if 'index_fa' in self.labels_dict:
                index_cell_line = self.labels_dict['index_fa'][cell_line]
                data_fa = self.data_dict['fa'].iloc[index_cell_line].reset_index(drop=True)
            else:
                data_fa = self.data_dict['fa']    
            assert (data_fa.shape[0] ==  len(self.labels_dict[cell_line]))

            return kf, data_fa.copy(), self.labels_dict[cell_line].copy()

        else:

            return kf, self.data_dict[cell_line].copy(), self.labels_dict[cell_line].copy()


    def set_labels_value(self):

        for key in self.labels_dict.keys():
                if key != 'fa':

                    correlated_pairs = spearman_corr(self.data_dict[key], self.spearman_corr_threshold, verbose=verbose)
                    self.data_dict[key] =  remove_correlated_features(self.data_dict[key], self.labels_dict[key], 
                                                                      correlated_pairs, type_test=type_test, verbose=verbose)
                    


    def return_data(self, 
                    cell_line, 
                    hyper_tuning=False, 
                    sequence=False,
                    random_state=123,
                    test_size=0.25, 
                    validation_size=0.15,
                    augmentation=False):
        
        """Splits data into training and test set for model testing. Splits further the training
        set and discard the test set if used for hyperparameters tuning.

        Parameters
        --------------
        cell_line (str): name of the cell line. Possible values are ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']
        hyper_tuning (bool): if True, creates a training and validation set
            and discards test set.
            Default: False
        sequence (bool): if the data passed are the genomic sequence.
            Default: False
        random_state (int): initial random seed for dataset split.
            Default: 123
        test_size (float): size of test set
            Default: 0.25
        validation_size (float): size of validation set
            Default: 0.15
        data_augmentation (bool): whether to apply or not data augmentation.
            Default: False

        Returns
        ---------------
        Dataframes of Training set, Test set, training labels, test labels, index (info about sequence)
        """
        
        if cell_line not in CELL_LINES:
            raise ValueError(
            f"Argument 'cell_line' has an incorrect value: use one among {CELL_LINES}")
            
        self.split_data(cell_line=cell_line, hyper_tuning=hyper_tuning, sequence=sequence, test_size=test_size, 
                        validation_size=validation_size, random_state=random_state)
    
        if augmentation:
            self.X_train, self.y_train = data_augmentation(self.X_train, self.y_train, 
                                                           sequence=sequence, threshold=0.15)

        return ( self.X_train, self.X_test,
                self.y_train, self.y_test)              

                



class Dataset_Wrap(Dataset):
    """Builds a Dataset object and saves indexes of positive and negative labels.
    If the data are the genomic sequence, transforms labels in integer and applies MICE imputer.

    Attributes:
    ---------------
    pos_index: indexes of positive labels.
    neg_index: indexes of negative labels.
    """
    def __init__(self, X, y, sequence=False):
        super(Dataset_Wrap, self).__init__()
        
        self.X = X
        self.y = y
        self.sequence = sequence
        
        self.pos_index = list(self.X[self.y==1].index)
        self.neg_index = list(self.X[self.y==0].index)
        
        # fit one-hot encoder
        self.onehot_encoder = OneHotEncoder(sparse=False).fit(np.array(['t', 'g', 'c', 'a']).reshape(-1, 1)) 

        # select with equal probability one of the nucleotides

    def __len__(self):
        return (self.X.shape[0])  

        
    def __getitem__(self, i):

        data = self.X.iloc[i]  
        
        if self.sequence:
            # all the letters in lowercase
            data = list(data.lower()) 
            # value n corresponds to nan, so we substitute it with a random bp
            bp = random.choice(['a','c','g','t'])
            data = [bp if i =='n' else i for i in data] # CHECK!
            # one hot encode
            data = self.onehot_encoder.transform(np.array(data).reshape(-1, 1))

            data = data.T
            
        data = torch.tensor(data)

        if isinstance(self.y, pd.Series):
            self.y = np.array(self.y)
        label = torch.tensor([self.y[i].astype(int)]).reshape(-1)

        return data.to(device), label.to(device)





class BalancePos_BatchSampler(Sampler):
    """Sampler that evenly distributes positive observations among
    all batches.
    """
    def __init__(self, dataset, batch_size, random_state=123):
        
        self.pos_index = dataset.pos_index
        self.neg_index = dataset.neg_index
        self.random_state = random_state
        
        self.batch_size = batch_size
        
        if len(dataset) % self.batch_size >0:
            self.n_batches = (len(dataset) // self.batch_size) +1
        else:
            self.n_batches = len(dataset) // self.batch_size
    
    def __iter__(self):
        random.seed(self.random_state)
        random.shuffle(self.pos_index)
        random.shuffle(self.neg_index)
        
        # create chunks of positive and negative labels
        # chunks of positive labels
        pos_batches  = np.array_split(self.pos_index, self.n_batches+1)
        # chunks of negative labels
        neg_batches = np.array_split(self.neg_index, self.n_batches+1)
        neg_batches.reverse()
        
        # create batches of size 100 with the same number of positive obs. in this way we are guaranteed to have
        #evenly distributed positive observations across all the batches, so we can always apply SMOTE 
        balanced = [ np.concatenate((p_batch, n_batch)).tolist() for p_batch, n_batch in zip(pos_batches, neg_batches) ]
        random.shuffle(balanced)
        return iter(balanced)
    
    def __len__(self):
        return self.n_batches




class Build_DataLoader_Pipeline():
    """Preprocesses data and builds a dataloader object.

    Applies robust scaler to genomic features and one hot encoding to genomic sequence.
    Applies MICE imputer.
    Removes correlated features and features uncorrelated with the target.
    Saves the object containing Data_Prepare class with preprocessed data.
    Splits dataset in training and test set. If the usage is for hyperparameters tuning, 
    splits the training set further and discard the test set and returns data of the selected 
    cell line.
    Creates dataloader object with evenly distributed positive targets in all the batches.

    Parameters
    ------------------
    data_dict (dict): dictionary of cell lines (pd.DataFrame).
    labels_dict (dict): dictionary of cell lines labels (pd.Series).
    path_name (str): name of pickle file storing the Data_Prepare class 
    containing preprocessed data.
    type_test (str or list of str): type of test. Values are ['kruskal_wallis_corr', 'wilcoxon_corr'].
    intersection (bool): whether to remove the uncorrelated features selected by all the methods (intersection)
        or the uncorrelated features selected by at least one method (union).
        Default: False
    pb_corr_threshold: point-biserial correlation threshold for correlation
        between X and y.
        Default: 0.05
    kruskal_pval_threshold: kruskal-wallis p-value threshold for correlation
        between X and y.
        Default: 0.05
    wilcoxon_pval_threshold: wilcoxon signed-rank p-value threshold for
        correlation between X and y.
        Default: 0.05
    spearman_corr_threshold: Spearman correlation threshold for correlation
        between different features.
        Default: 0.85
    verbose (bool): returns info.
        Default: False
    """

    def __init__(self,
                 data_dict=None, 
                 labels_dict=None,
                 path_name=None,
                 type_test='kruskal_wallis_test',
                 intersection=False, 
                 pb_corr_threshold=0.05,
                 kruskal_pval_threshold = 0.05,
                 wilcoxon_pval_threshold = 0.05,
                 spearman_corr_threshold=0.85,
                 verbose=False):
    
        self.data_dict =  data_dict
        self.labels_dict = labels_dict
        self.path_name = path_name
        self.type_test = type_test
        self.intersection = intersection
        self.pb_corr_threshold = pb_corr_threshold
        self.kruskal_pval_threshold = kruskal_pval_threshold
        self.wilcoxon_pval_threshold = wilcoxon_pval_threshold
        self.spearman_corr_threshold = spearman_corr_threshold

        self.verbose = verbose

        self.data_class = []
        self.transform = False
        self.correlation_with_label = False
        self.correlation_btw_features = False
    
        
        # if there exists already the class of preprocessed data, load it
        # else instantiate it and do all the preprocessing
        if os.path.exists("data_prepare_class_{}".format(self.path_name)):
            with open("data_prepare_class_{}".format(self.path_name), "rb") as fin:
                self.data_class = pickle.load(fin)
        else:
            self.data_class = Data_Prepare(self.data_dict, self.labels_dict)
            self.data_class.transform()
            print('Data transformation Done!\n')
            self.data_class.correlation_with_label(type_test=self.type_test, intersection=self.intersection, verbose=self.verbose)
            print('Check correlation with labels Done!\n')
            self.data_class.correlation_btw_features(verbose=self.verbose)  
            print('Check correlation between features Done!\n')  
            
            with open(f"data_prepare_class_{self.path_name}", "wb") as fout:
                pickle.dump(self.data_class, fout)
        
        print('Data Preprocessing Done!')
            
        
        
    def return_data(self, 
                    cell_line, 
                    hyper_tuning=False, 
                    sequence=False,
                    random_state=123,
                    augmentation=False,
                    test_size=0.25, 
                    validation_size=0.15,
                    batch_size = 100):
        """
        Builds DataLoader object.

        Parameters
        --------------
        cell_line (str): name of the cell line. Possible values are ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']
        hyper_tuning (bool): if True, creates a training and validation set
            and discards test set.
            Default: False
        sequence (bool): if the data passed are the genomic sequence.
        test_size (float): size of test set
        validation_size (float): size of validation set

        Returns
        ---------------
        DataLoader of training set, DataLoader of test set

        """

        # retrieve the data 
        X_train, X_test, y_train, y_test = self.data_class.return_data(cell_line=cell_line, 
                                                                          hyper_tuning=hyper_tuning, 
                                                                          sequence=sequence,
                                                                          random_state=random_state,
                                                                          test_size=test_size,
                                                                          validation_size=validation_size,
                                                                          augmentation=augmentation)
        
        train_wrap = Dataset_Wrap(X_train, y_train, sequence=sequence)
        test_wrap = Dataset_Wrap(X_test, y_test, sequence=sequence)
        
        loader_train = DataLoader(dataset = train_wrap, 
                                  batch_sampler = BalancePos_BatchSampler(train_wrap, batch_size= batch_size))
        loader_test = DataLoader(dataset = test_wrap, batch_size= batch_size*2, shuffle=True,
                                    generator = g_gpu.manual_seed(random_state+30))
            


        return  loader_train, loader_test


