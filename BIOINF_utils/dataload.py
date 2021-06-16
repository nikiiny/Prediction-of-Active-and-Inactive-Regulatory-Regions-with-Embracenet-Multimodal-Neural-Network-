import os 
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
import random
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import itertools
from scipy.stats import pointbiserialr
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from torch.utils.data import Sampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Load_Create_Task():
    """Loads enhancers and promoters data, creates 2 dictionaries for each and 
    stores the different data about cell lines and labels.
    Returns the selected task among: 'active_E_vs_inactive_E', 'active_P_vs_inactive_P', 
    'active_E_vs_active_P', 'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_all'.
    
    Parameters
    ----------------
    directory (str): directory where data are stored
    
    Returns
    ----------------
    Dictionary of data, Dictionary of labels
    """
    def __init__(self, directory='data'):
        self.directory = directory
        
        self.enhancers_dict = []
        self.promoters_dict = []
        
        self.enhancers_labels_dict = {}
        self.promoters_labels_dict = {}
        

    def data_loader(self, file_in_dir):
        """Loads enhancers and promoters data. Stores the different files for
        each in a dictionary and calls them with the same names to simplify
        files retrieval.
        
        Parameters
        ------------
        file_in_dir (str): names of enhancers/promoters files
        
        Returns
        ------------
        Dictionary
        """
    
        data = {}

        for file in tqdm(file_in_dir, desc='Loading data'):
            if file.endswith('.csv'):
                name = re.search('data/.*/(.*).csv', file).group(1)
                name = re.sub('-','',name)
                data[name.upper()] = pd.read_csv(file)

            elif file.endswith('.bed'):
                name = re.search('data/.*/.*(bed)', file).group(1)
                data[name] = pd.read_csv(file, sep='\t')

            elif file.endswith('.fa'):
                name = re.search('data/.*/.*(fa)', file).group(1)

                fa = pd.DataFrame()
                with open(file) as FILE:
                    fa['sequence'] = [line.strip() for i,line in enumerate(FILE) if i%2==0]
                    FILE.close()
                with open(file) as FILE:
                    fa['chromosome'] = [line.strip() for i,line in enumerate(FILE) if i%2!=0]
                    FILE.close()

                data[name] = fa
                data[name][['0','chrom','chromStart','chromEnd']] = data[name]['sequence'].str.split('>|:|-', expand=True)
                data[name] = data[name].drop(['sequence','0'],axis=1)

        return data
    
    def load(self, verbose=False) :
        """Loads enhancers and promoters data and creates dictionary of data
        and labels.
        
        Parameters:
        ------------
        verbose (bool): prints data names and shape
        """

        enhancers = []
        for PATH in os.listdir( os.path.join(self.directory, 'enhancers') ):
            enhancers.append(os.path.join(self.directory, 'enhancers', PATH))

        promoters = []
        for PATH in os.listdir( os.path.join(self.directory, 'promoters') ):
            promoters.append(os.path.join(self.directory, 'promoters', PATH))
            
        self.enhancers_dict = self.data_loader(enhancers)
        self.promoters_dict = self.data_loader(promoters)
        
        for key in self.enhancers_dict.keys():
            if key not in ['fa','bed']:
                self.enhancers_labels_dict[key] = self.enhancers_dict['bed'][key]
                self.promoters_labels_dict[key] = self.promoters_dict['bed'][key]
        
        if verbose:
            print('enhancers files:\n{}\n'.format(sorted(self.enhancers_dict.keys())))
            for key in self.enhancers_dict.keys():
                print('{} has shape: {}'.format(key, self.enhancers_dict[key].shape))

            print('\npromoters files:\n{}\n'.format(sorted(self.enhancers_dict.keys())))
            for key in self.promoters_dict.keys():
                print('{} has shape: {}'.format(key, self.promoters_dict[key].shape))
        
    
    def get_task(self, task):
        """Returns selected task among:
        - active enhancers vs inactive enhancers
        - active promoters vs inactive promoters
        - active enhancers vs active promoters
        - inactive enhancers vs inactive promoters
        - active enhancers + active promoters vs rest
        
        Parameters:
        ------------
        task (str): 'active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 
        'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_rest'.
        
        Returns:
        ------------
        Dictionary of data, Dictionary of label
        """
        
        if task not in ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 
                        'active_E_vs_active_P', 'inactive_E_vs_inactive_P',
                        'active_EP_vs_inactive_rest']:
            raise ValueError(
                "Argument 'task' has an incorrect value: use 'active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 'inactive_E_vs_inactive_P','active_EP_vs_inactive_rest' ")
    
        
        if task == 'active_E_vs_inactive_E':
            if 'bed' in self.enhancers_dict:
                del self.enhancers_dict['bed']
            return self.enhancers_dict, self.enhancers_labels_dict

        
        elif task == 'active_P_vs_inactive_P':
            if 'bed' in self.promoters_dict:
                del self.promoters_dict['bed']
            return self.promoters_dict, self.promoters_labels_dict
    
    
        elif task == 'active_EP_vs_inactive_rest':
            data_dict = {}
            data_labels_dict = {}
            
            for key in self.enhancers_dict.keys():
                if key != 'bed':
                    data_dict[key] = pd.concat([self.enhancers_dict[key], self.promoters_dict[key]])
                    if key != 'fa':
                        data_labels_dict[key] = pd.Series( np.concatenate(( self.enhancers_labels_dict[key], self.promoters_labels_dict[key] )) )
            
            return data_dict, data_labels_dict
        
        
        elif task == 'active_E_vs_active_P':
            data_dict = {}
            data_labels_dict = {}
            # create a dictionary to store the index of labels in order to
            #select the correct observations of fasta file later on
            data_labels_dict['index_fa'] = {}
            
            for key in self.enhancers_dict.keys():
                if key not in ['bed', 'fa']:
                    
                    data = pd.concat([ self.enhancers_dict[key], self.promoters_dict[key] ])
                    # to retrieve index
                    original_labels = pd.Series( np.concatenate(( 
                                        self.enhancers_labels_dict[key], 
                                        self.promoters_labels_dict[key] )) )
                    # enhancers have label 1, promoters have label 0  
                    new_labels = pd.Series( 
                            np.concatenate(( np.repeat(1, self.enhancers_dict[key].shape[0]), 
                                            np.repeat(0, self.promoters_dict[key].shape[0]) ))
                        )
                    # select only active enhancers and active promoters
                    index = original_labels[original_labels==1].index
                    data_dict[key] = data.iloc[index].reset_index(drop=True)
                    data_labels_dict[key] = new_labels.iloc[index].reset_index(drop=True)
                    # store index of active enhancers and active promoters for every cell line
                    #in order to retrieve later on the correct fa observations
                    data_labels_dict['index_fa'][key] = index
                    
                    assert ( len(data_labels_dict[key]) == data_dict[key].shape[0] == len(data_labels_dict['index_fa'][key]) )
                        
            data_dict['fa'] = pd.concat([ self.enhancers_dict['fa'], self.promoters_dict['fa'] ]) 
            
            return data_dict, data_labels_dict
        
        
        elif task == 'inactive_E_vs_inactive_P':
            data_dict = {}
            data_labels_dict = {}
            # create a dictionary to store the index of labels in order to
            #select the correct observations of fasta file later on
            data_labels_dict['index_fa'] = {}
            
            for key in self.enhancers_dict.keys():
                if key not in ['bed', 'fa']:
                    
                    data = pd.concat([ self.enhancers_dict[key], self.promoters_dict[key] ])
                    # to retrieve index
                    original_labels = pd.Series( np.concatenate(( 
                                        self.enhancers_labels_dict[key], 
                                        self.promoters_labels_dict[key] )) )
                    # enhancers have label 1, promoters have label 0
                    new_labels = pd.Series( 
                            np.concatenate(( np.repeat(1, self.enhancers_dict[key].shape[0]), 
                                            np.repeat(0, self.promoters_dict[key].shape[0]) ))
                        )
                    
                    # select only inactive enhancers and inactive promoters
                    index = original_labels[original_labels==0].index
                    data_dict[key] = data.iloc[index]
                    data_labels_dict[key] = new_labels.iloc[index]
                    # store index of inactive enhancers and inactive promoters for every cell line
                    #in order to retrieve later on the correct fa observations
                    data_labels_dict['index_fa'][key] = index
                    
                    assert ( len(data_labels_dict[key]) == data_dict[key].shape[0] == len(data_labels_dict['index_fa'][key]) )
                        
            data_dict['fa'] = pd.concat([ self.enhancers_dict['fa'], self.promoters_dict['fa'] ]) 
            
            return data_dict, data_labels_dict



class Data_Preparation():
    
    def __init__(self, data_dict, labels_dict, n_neighbors=5):
        
        self.labels_dict = labels_dict.copy()
        self.index = data_dict['H1'][['chrom','chromStart','chromEnd','strand']].copy()
        self.data_dict = data_dict.copy()
        
        self.data_dict['fa'] = self.data_dict['fa']['chromosome']
        
        # drop observations info
        for key in self.data_dict.keys():
            if key != 'fa':
                self.data_dict[key] = self.data_dict[key].drop(['chrom','chromStart','chromEnd','strand'], axis=1)
                
        
        self.n_neighbors = n_neighbors
        
        self.robust_scaler = RobustScaler()
      #  self.label_encoder = LabelEncoder()
       # self.onehot_encoder = OneHotEncoder(sparse=False) 
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
        self.sequence = []
        
        self.to_drop = dict()
    
                    
    def scale_data_genfeatures(self):
        for key in self.data_dict.keys():
            if key != 'fa':
                self.data_dict[key] = pd.DataFrame(self.robust_scaler.fit_transform(self.data_dict[key].values),
                                                   index=self.data_dict[key].index, 
                                                   columns=self.data_dict[key].columns)
                
    
    def knn_imputation_genfeatures(self):
        for key in self.data_dict.keys():
            if key != 'fa':
                self.data_dict[key] = pd.DataFrame(self.knn_imputer.fit_transform(self.data_dict[key].values),
                                                   index=self.data_dict[key].index, 
                                                   columns=self.data_dict[key].columns)

    
    def transform(self):
        
        self.scale_data_genfeatures()
        self.knn_imputation_genfeatures()
        
  

    def point_biserial_corr(self, X, y, verbose=False):
        """Point biserial correlation returns the correlation between a continuous and
        binary variable (target). It is a parametric test, so it assumes the data to be normally
        distributed."""
        
        uncorrelated = set()

        for col in X.columns:
            x = X[col]
            corr,_ = pointbiserialr(x,y)
            if abs(corr) < 0.05:
                uncorrelated.add(col)
                if verbose:
                    print('column: {}, Point-biserial Correlation: {}'.format(col, round(corr,4)))
        
        return uncorrelated
    
    
    def logistic_regression_corr(self, X, y, verbose=False):
        """The Logistic regression can be used to assess if a continuous variable has any 
         effect on the target. It doesn't assume anything about the distribution of the variables.
         The metric used is the area under the precision recall curve, which """
        
        uncorrelated = set()
        
        for col in X.columns:
            x = X[col].values.reshape(-1, 1)

            # perform 3-folds cv with logistic regression
            cv = KFold(n_splits=3, random_state=123, shuffle=True)
            model = LogisticRegression()

            baseline_binary_pos = len(y[y==1]) / len(y)

            # compute the AUPRC for the positive score
            AUPRC = make_scorer(average_precision_score, average='weighted')
            scores = cross_val_score(model, x, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")
        
            if scores.mean() <= baseline_binary_pos:
                uncorrelated.add(col)
                
                if verbose:
                    print('column: {}, AUPRC: {}, Baseline positive class: {}'.format(col, round(scores.mean(),4), round(baseline_binary_pos,4)))
        
        return uncorrelated

    
    def correlation_with_label(self, type_corr='all', verbose=False):
        
        if type_corr not in ['point_biserial_corr', 'logistic_regression', 'all']:
            raise ValueError(
            "Argument 'type_corr' has an incorrect value: use 'point_biserial_corr', 'logistic_regression', 'all'")
            
        
        if type_corr == 'logistic_regression':
            for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print(key)
                    self.to_drop[key] = self.logistic_regression_corr(self.data_dict[key], self.labels_dict[key], verbose=verbose)
                    # remove uncorrelated features
                    if self.to_drop[key]:
                        self.data_dict[key] = self.data_dict[key].drop(list(self.to_drop[key]), axis=1)
        
        
        elif type_corr == 'point_biserial_corr':    
            for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print(key)
                    self.to_drop[key] = self.point_biserial_corr(self.data_dict[key], self.labels_dict[key], verbose=verbose)
                    # remove uncorrelated features
                    if self.to_drop[key]:
                        self.data_dict[key] = self.data_dict[key].drop(list(self.to_drop[key]), axis=1)
        
        
        elif type_corr == 'all':
             for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print(key)
                    self.to_drop[key] = self.point_biserial_corr(self.data_dict[key], self.labels_dict[key], verbose=verbose)
                    self.to_drop[key].intersection(self.logistic_regression_corr(self.data_dict[key], self.labels_dict[key], verbose=verbose))
                    # remove uncorrelated features
                    if self.to_drop[key]:
                        self.data_dict[key] = self.data_dict[key].drop(list(self.to_drop[key]), axis=1)
        
    
    
    def spearman_corr(self, X, verbose=False):
        """Spearman correlation checks for linear correlation between continuous features.
        It is non-parametric, so normality of the variables is not necessary."""
        
        correlated = set()
        
        for col1, col2 in itertools.combinations(X.columns, 2):
            corr, _ = spearmanr(X[col1].values, X[col2].values)
            
            if corr >= 0.85:
                correlated.add(frozenset({col1,col2}))
                
                if verbose:
                    print('correlated columns: {} - {}, Spearman Correlation {}'.format(col1, col2, round(corr,4)))
        
        return correlated
    
    
    

    def remove_correlated_feature(self, X, y, correlated_pairs, verbose=False):

        correlated_to_remove = set()

        for pair in correlated_pairs:
            col1, col2 = pair
            x1 = X[col1].values.reshape(-1, 1)
            x2 = X[col2].values.reshape(-1, 1)

            # perform 3-folds cv with logistic regression
            cv = KFold(n_splits=3, random_state=123, shuffle=True)
            model = LogisticRegression()

            # compute the AUPRC for the positive score
            AUPRC = make_scorer(average_precision_score, average='weighted')
            scores1 = cross_val_score(model, x1, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")
            scores2 = cross_val_score(model, x1, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")

            if verbose:
                print('columns to compare: {} vs {}, AUPRC: {} vs {}'.format(col1, col2, scores1.mean(), scores2.mean()))

            if scores1.mean() >= scores2.mean():
                correlated_to_remove.add(col2)
            else:
                correlated_to_remove.add(col1)
    
    
    
    def correlation_btw_features(self, verbose=False):
        
        self.to_drop = dict()
        
        for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print(key)
                    correlated_pairs = self.spearman_corr(self.data_dict[key], verbose=verbose)
                    self.to_drop[key] =  self.remove_correlated_feature(self.data_dict[key], self.labels_dict[key], correlated_pairs, verbose=verbose)
                    # for each pair of correlated features, remove the one less correlated to the output
                    if self.to_drop[key]:
                        self.data_dict[key] = self.data_dict[key].drop(list(self.to_drop[key]), axis=1)
         
    
    def split_data(self, cell_line, test_size, validation_size):
        
        if self.sequence:
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
                                                                                 random_state=456, shuffle=True) 

            if self.hyper_tuning:
                assert (self.X_train.shape[0] ==  len(self.y_train))
                    
                self.X_train, self.X_test,  self.y_train, self.y_test = train_test_split(self.X_train, 
                                                                                     self.y_train,
                                                                                     test_size=validation_size,
                                                                                     random_state=123, shuffle=True) 
                   
            
        else:
            assert (self.data_dict[cell_line].shape[0] ==  len(self.labels_dict[cell_line]))

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_dict[cell_line], 
                                                                                    self.labels_dict[cell_line],
                                                                                    test_size=test_size, 
                                                                                    random_state=123, shuffle=True) 
            
            if self.hyper_tuning: 
                assert (self.X_train.shape[0] ==  len(self.y_train))

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, 
                                                                                        self.y_train,
                                                                                        test_size=validation_size, 
                                                                                        random_state=456, shuffle=True) 
        
    
    def return_data(self, cell_line, hyper_tuning=False, sequence=True, test_size=0.25, validation_size=0.15):
    
        self.sequence=sequence
        self.hyper_tuning=hyper_tuning
        
        if cell_line not in ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']:
            raise ValueError(
            "Argument 'cell_line' has an incorrect value: use 'A549', 'GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7'")
            
        
        self.split_data(cell_line=cell_line, test_size=test_size, validation_size=validation_size)
    

        return ( self.X_train.reset_index(drop=True), self.X_test.reset_index(drop=True), 
                self.y_train.reset_index(drop=True) , self.y_test.reset_index(drop=True),
                self.index )
                



class Dataset_Wrap(Dataset):
    def __init__(self, X, y, n_neighbors, sequence=False):
        super(Dataset_Wrap, self).__init__()
        
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
        self.sequence = sequence
        
        self.pos_index = list(self.X[self.y==1].index)
        self.neg_index = list(self.X[self.y==0].index)
        
        # some observations don't have some of the nucleotides, so it will result in
        #matrices of different dimensions which cannot be concatenated
        self.label_encoder = LabelEncoder().fit(np.array(['a','c','g','n','t']))
        # get rid of value 3 which is 'n' (nan)
        self.onehot_encoder = OneHotEncoder(sparse=False).fit(np.array([0,1,2,4]).reshape(-1, 1)) 
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)


    def __len__(self):
        return (self.X.shape[0])  

        
    def __getitem__(self, i):

        data = self.X.iloc[i]  
        
        if self.sequence:
            # all the letters in lowercase
            data = [i for i in data.lower()]
            
            # apply encoding
            data = self.label_encoder.transform(data)
            # value 3 corresponds to n (nan)
            data = [np.nan if i ==3 else i for i in data]
            
            # impute missing data with knn
            data =  self.knn_imputer.fit_transform( np.array(data).reshape(-1,1) ).astype(int).round()
            
            # one hot encode data
          #  data = self.onehot_encoder.transform(data)
        
        data = torch.tensor(data)
            
       # if self.sequence: 
            # (channels, size of the matrix)
        #    data = data.reshape(1, data.shape[0], data.shape[1])

        label = torch.tensor([self.y[i].astype(int)]).reshape(-1)

        return data.to(device), label.to(device)





class BalancePos_BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        
        self.pos_index = dataset.pos_index
        self.neg_index = dataset.neg_index
        
        self.batch_size = batch_size
        
        if len(dataset) % self.batch_size >0:
            self.n_batches = (len(dataset) // self.batch_size) +1
        else:
            self.n_batches = len(dataset) // self.batch_size
    
    def __iter__(self):
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


import pickle
from varname import nameof




class Build_DataLoader_Pipeline():

    def __init__(self,
                 data_dict, 
                 labels_dict,
                 path_name=None,
                 n_neighbors=5,
                 type_corr='all',
                 verbose=False):
    
        self.data_dict =  data_dict
        self.labels_dict = labels_dict
        self.path_name = path_name
        self.n_neighbors = n_neighbors
        self.type_corr = type_corr
        self.verbose = verbose

        self.data_class = []
        self.transform = False
        self.correlation_with_label = False
        self.correlation_btw_features = False
        
        self.index = []
    
        
        # if there exists already the class of preprocessed data, load it
        # else instantiate it and do all the preprocessing
        if os.path.exists("data_preparation_class_{}".format(self.path_name)):
            with open("data_preparation_class_{}".format(self.path_name), "rb") as fin:
                self.data_class = pickle.load(fin)
        else:
            self.data_class = Data_Preparation(self.data_dict, self.labels_dict, n_neighbors=self.n_neighbors)
            self.data_class.transform()
            print('Data transformation Done!\n')
         #   self.data_class.correlation_with_label(type_corr=self.type_corr, verbose=self.verbose)
          #  print('Check correlation with labels Done!\n')
           # self.data_class.correlation_btw_features(verbose=self.verbose)  
            #print('Check correlation between features Done!\n')  
            
            with open("data_preparation_class_{}".format(self.path_name), "wb") as fout:
                pickle.dump(self.data_class, fout)
        
        print('Data Preprocessing Done!')
            
        
        
    def return_data(self, 
                    cell_line, 
                    hyper_tuning=False, 
                    sequence=False,
                    test_size=0.25, 
                    validation_size=0.15,
                    batch_size = 100):
            
        # retrieve the data 
        X_train, X_test, y_train, y_test, index = self.data_class.return_data(cell_line=cell_line, 
                                                                          hyper_tuning=hyper_tuning, 
                                                                          sequence=sequence,
                                                                          test_size=test_size,
                                                                          validation_size=validation_size)
        
        self.index = index
        
        train_wrap = Dataset_Wrap(X_train, y_train, sequence=sequence, n_neighbors=self.n_neighbors)
        test_wrap = Dataset_Wrap(X_test, y_test, sequence=sequence, n_neighbors=self.n_neighbors)
        
        loader_train = DataLoader(dataset = train_wrap, 
                                  batch_sampler = BalancePos_BatchSampler(train_wrap, batch_size= batch_size))
        loader_test = DataLoader(dataset = test_wrap, 
                                 batch_sampler = BalancePos_BatchSampler(test_wrap, batch_size= batch_size*2)) 

        return  loader_train, loader_test
