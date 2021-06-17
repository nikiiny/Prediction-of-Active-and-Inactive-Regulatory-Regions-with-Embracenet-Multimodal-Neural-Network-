import os 
from tqdm.notebook import tqdm
import re
import pandas as pd
import numpy as np
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
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Data_Prepare():
    """Applies robust scaler and KNN imputer to genomic features. 
    Checks for highly correlated features (Spearman correlation) and features not 
    correlated to the output (Point Biserial correlation or/and Logistic regression AUPRC).
    Splits dataset either for hyperparameters tuning or model testing.

    Parameters
    ---------------
    data_dict (dict): dictionary of cell lines (pd.DataFrame).
    labels_dict (dict): dictionary of cell lines labels (pd.Series).
    n_neighbours: number of neighbours for KNN imputer.
        Default: 5
    """
    
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
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
        self.sequence = []
        
        self.to_drop = dict()
    
                    
    def scale_data_genfeatures(self):
        """Applies robust scaler to numeric genomic features by row/column?"""
        for key in self.data_dict.keys():
            if key != 'fa':
                self.data_dict[key] = pd.DataFrame(self.robust_scaler.fit_transform(self.data_dict[key].values),
                                                   index=self.data_dict[key].index, 
                                                   columns=self.data_dict[key].columns)
                
    
    def knn_imputation_genfeatures(self):
        """Applies KNN imputation to numeric genomic features."""
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
        distributed.
        
        Parameters
        ----------------
        X (pd.DataFrame): data.
        y (pd.Series): label.
        verbose (bool): returns scores.
            Default: False

        Returns
        ------------
        Set of columns uncorrelated with the target.
        """
        
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
         The metric used is the area under the precision recall curve, which is compared to the 
         baseline (when the model guesses at random the target.

        Parameters
        ----------------
        X (pd.DataFrame): data.
        y (pd.Series): label.
        verbose (bool): returns scores.
            Default: False

        Returns
        ----------------
        Set of columns uncorrelated with the target (AUPRC = baseline).
        """
        
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
        """Checks correlation with label.

        Parameters
        ----------------
        type_corr (str): type of correlation. Values are ['point_biserial_corr', 'logistic_regression', 'all']
            Default: 'all'
        verbose (bool): returns info.
        """
        
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
        It is non-parametric, so normality of the variables is not necessary.

        Parameters
        ----------------
        X (pd.DataFrame): data.
        verbose (bool): returns scores.
            Default: False

        Returns
        ----------------
        List of pairs of highly correlated features (>0.85) in descending correlation order.
        """
        
        correlated = dict()
        
        for col1, col2 in itertools.combinations(X.columns, 2):
            corr, _ = spearmanr(X[col1].values, X[col2].values)
            
            if corr >= 0.85:
                correlated[corr]=[col1,col2]
                
                if verbose:
                    print('correlated columns: {} - {}, Spearman Correlation {}'.format(col1, col2, round(corr,4)))
        # order by descending correlation
        ord_correlated = OrderedDict(sorted(correlated.items(), reverse=True))
        
        # return list of correlated pairs in descending correlation order
        return list(ord_correlated.values())
    
    

    def remove_correlated_feature(self, X, y, correlated_pairs, verbose=False):
        """Removes the less correlated feature with the target from a pair of highly correlated feature.
        Correlation with the target is calculated as the AUPRC of a logistic regression between
        the feature and the target.

        Parameters
        ----------------
        X (pd.DataFrame): data.
        y (pd.Series): label
        correlated_pairs (list): list of pairs of correlated features
        verbose (bool): returns scores.
            Default: False

        Returns
        ----------------
        Dataframe with no correlated pairs.
        """

        for pair in correlated_pairs:
            col1, col2 = pair
            if col1 in X and col2 in X:
                x1 = X[col1].values.reshape(-1, 1)
                x2 = X[col2].values.reshape(-1, 1)

                # perform 3-folds cv with logistic regression
                cv = KFold(n_splits=3, random_state=123, shuffle=True)
                model = LogisticRegression()

                # compute the AUPRC for the positive score
                AUPRC = make_scorer(average_precision_score, average='weighted')
                scores1 = cross_val_score(model, x1, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")
                scores2 = cross_val_score(model, x2, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")

                if verbose:
                    print('columns to compare: {} vs {}, AUPRC: {} vs {}'.format(col1, col2, scores1.mean(), scores2.mean()))

                if scores1.mean() >= scores2.mean():
                    X = X.drop([col2], axis=1)
                    print('removed column: {}'.format(col2))
                else:
                    X = X.drop([col1], axis=1)
                    print('removed column:  {}'.format(col1))
                    
        # return new dataframe with dropped correlated features.
        return X
    
    
    
    def correlation_btw_features(self, verbose=False):
        """Runs Spearman correlation and remove less correlated feature from a pair
        of correlated features."""
        
        self.to_drop = dict()
        
        for key in self.data_dict.keys():
                if key != 'fa':
                    if verbose:
                        print('\n', key)
                    correlated_pairs = self.spearman_corr(self.data_dict[key], verbose=verbose)
                    self.data_dict[key] =  self.remove_correlated_feature(self.data_dict[key], self.labels_dict[key], correlated_pairs, verbose=verbose)
                    
                    
    
    def split_data(self, cell_line, hyper_tuning, sequence, test_size, validation_size):
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

        Returns
        ---------------
        Training set, Test set, training labels, test labels
        """
        
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

            if hyper_tuning:
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
            
            if hyper_tuning: 
                assert (self.X_train.shape[0] ==  len(self.y_train))

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, 
                                                                                        self.y_train,
                                                                                        test_size=validation_size, 
                                                                                        random_state=456, shuffle=True) 
        
    
    def return_data(self, cell_line, hyper_tuning=False, sequence=False, test_size=0.25, validation_size=0.15):
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
        test_size (float): size of test set
            Default: 0.25
        validation_size (float): size of validation set
            Default: 0.15

        Returns
        ---------------
        Dataframes of Training set, Test set, training labels, test labels, index (info about sequence)
        """
        
        if cell_line not in ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']:
            raise ValueError(
            "Argument 'cell_line' has an incorrect value: use 'A549', 'GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7'")
            
        
        self.split_data(cell_line=cell_line, hyper_tuning=hyper_tuning, test_size=test_size, validation_size=validation_size)
    

        return ( self.X_train.reset_index(drop=True), self.X_test.reset_index(drop=True), 
                self.y_train.reset_index(drop=True) , self.y_test.reset_index(drop=True),
                self.index )
                


                



class Dataset_Wrap(Dataset):
    """Builds a Dataset object and saves indexes of positive and negative labels.
    If the data are the genomic sequence, transforms labels in integer and applies KNN imputer.

    Attributes:
    ---------------
    pos_index: indexes of positive labels.
    neg_index: indexes of negative labels.
    """
    def __init__(self, X, y, n_neighbors=5, sequence=False):
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
            
        
        data = torch.tensor(data)
        label = torch.tensor([self.y[i].astype(int)]).reshape(-1)

        return data.to(device), label.to(device)





class BalancePos_BatchSampler(Sampler):
    """Sampler that evenly distributes positive observations among
    all batches.
    """
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




class Build_DataLoader_Pipeline():
    """Preprocesses data and builds a dataloader object.

    Applies robust scaler to genomic features and one hot encoding to genomic sequence.
    Applies KNN imputer.
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
    n_neighbours: number of neighbours for KNN imputer.
        Default: 5
    type_corr (str): type of correlation. Values are ['point_biserial_corr', 'logistic_regression', 'all']
        Default: 'all'
    verbose (bool): returns info.
        Default: False
    """

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
        if os.path.exists("data_prepare_class_{}".format(self.path_name)):
            with open("data_prepare_class_{}".format(self.path_name), "rb") as fin:
                self.data_class = pickle.load(fin)
        else:
            self.data_class = Data_Prepare(self.data_dict, self.labels_dict, n_neighbors=self.n_neighbors)
            self.data_class.transform()
            print('Data transformation Done!\n')
            self.data_class.correlation_with_label(type_corr=self.type_corr, verbose=self.verbose)
            print('Check correlation with labels Done!\n')
            self.data_class.correlation_btw_features(verbose=self.verbose)  
            print('Check correlation between features Done!\n')  
            
            with open("data_prepare_class_{}".format(self.path_name), "wb") as fout:
                pickle.dump(self.data_class, fout)
        
        print('Data Preprocessing Done!')
            
        
        
    def return_data(self, 
                    cell_line, 
                    hyper_tuning=False, 
                    sequence=False,
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
