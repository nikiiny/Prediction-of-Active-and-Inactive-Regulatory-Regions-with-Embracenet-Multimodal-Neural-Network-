import os 
from tqdm.notebook import tqdm
import re
import pandas as pd
import numpy as np


TASKS = ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 
                        'active_E_vs_active_P', 'inactive_E_vs_inactive_P',
                        'active_EP_vs_inactive_rest']

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
            print(f'enhancers files:\n{sorted(self.enhancers_dict.keys())}\n')
            for key in self.enhancers_dict.keys():
                print(f'{key} has shape: {self.enhancers_dict[key].shape}')

            print(f'\npromoters files:\n{sorted(self.enhancers_dict.keys())}\n')
            for key in self.promoters_dict.keys():
                print(f'{key} has shape: {self.promoters_dict[key].shape}')
        
    
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
        
        if task not in TASKS:
            raise ValueError(
                f"Argument 'task' has an incorrect value: use one among {TASKS} ")
    
        
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

