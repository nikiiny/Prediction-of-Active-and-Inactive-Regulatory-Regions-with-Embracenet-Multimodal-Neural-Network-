import pandas as pd
import numpy as np
import os
import re
import pickle
#from tqdm.auto import tqdm
from tqdm.auto import tqdm
import sqlite3
from sqlalchemy import create_engine
from collections import defaultdict
import copy
from sklearn.model_selection import train_test_split

import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim 
import timm.optim
from torch.utils.data import Dataset, DataLoader
import optuna
import botorch
from optuna.integration import BoTorchSampler
from optuna.samplers import TPESampler, RandomSampler

from .utils import (EarlyStopping, F1_precision_recall, AUPRC, size_out_convolution, 
    weight_reset, get_loss_weights_from_dataloader, get_loss_weights_from_labels, 
    get_input_size, plot_other_scores, plot_F1_scores, save_best_model)
from BIOINF_tesi.data_pipe.utils import data_augmentation
from BIOINF_tesi.data_pipe.dataprepare import Data_Prepare, Dataset_Wrap, BalancePos_BatchSampler



def fit(model, 
        train_loader, 
        test_loader, 
        criterion,
        device,
        optimizer=None, 
        num_epochs=100,
        patience=5,
        delta=0,
        verbose=True): 
    
    """Performs the training of the model. It implements also early stopping
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    criterion: loss function for training the model.
    device: 'CPU' or 'CUDA'.
    optimizer (torch.optim): optimization algorithm for training the model. 
    num_epochs (int): number of epochs.
    patience (int): number of epochs in which the test error is not anymore decreasing
        before stopping the training.
        Default:5
    delta (int): minimum decrease in the test error to continue with the training.
        Default:0
    verbose (bool): prints the training error, test error, F1 training score, F1 test score 
        at each epoch.
        Default: True
    
    Attributes:
    ------------------
    f1_train_scores: stores the F1 training scores for each epoch.
    f1_test_scores: stores the F1 test scores for each epoch.
    
    Returns:
    ------------------
    Lists of F1 training scores and F1 test scores at each epoch.
    Prints training error, test error, F1 training score, F1 test score at each epoch.
    """

    basepath = 'exp'

    # keep track of epoch losses 
    AUPRC_train_scores = []
    AUPRC_test_scores = []
    F1_precision_recall_test_scores = []

    # convert model data type to double
    model = model.double().to(device)

    # define early stopping
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    
    
    for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
        train_loss = 0.0
        test_loss = 0.0
        
        AUPRC_train = 0.0
        AUPRC_test = 0.0
        F1_precision_recall_test = np.zeros(3)

    
    # set the model in training modality
        model.train()

        for data, target in train_loader:
        
            target.to(device)
            data.to(device)
            
            target = target.reshape(-1)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.double())
            # calculate the batch loss as the sum of all the losses
            try:
                loss = criterion.double()(output.float(), target.squeeze()) 
            except:
                loss = criterion.float()(output.float(), target.squeeze()) 
            # backward pass: compute gradient of the loss wrt model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()
            # calculate AUPRC training score as a weighted sum of the single AUPRC scores
            AUPRC_train += AUPRC(output,target)

        
        # set the model in testing modality
        model.eval()
        for data, target in test_loader:
            
            target.to(device)
            data.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.double())
            # calculate the batch loss as the sum of all the losses
            try:
                loss = criterion.double()(output.float(), target.squeeze()) 
            except:
                loss = criterion.float()(output.float(), target.squeeze()) 
            # update test loss
            test_loss += loss.item()
            # calculate AUPRC test score as a weighted sum of the single AUPRC scores
            AUPRC_test += AUPRC(output,target) 
            F1_precision_recall_test += F1_precision_recall(output,target)
        
    
        # calculate epoch score by dividing by the number of observations
        AUPRC_train /= (len(train_loader))
        AUPRC_test /= (len(test_loader))
        F1_precision_recall_test /= (len(test_loader))
        # store epoch score
        AUPRC_train_scores.append(AUPRC_train)    
        AUPRC_test_scores.append(AUPRC_test)
        F1_precision_recall_test_scores.append(F1_precision_recall_test)
          
        # print training/test statistics 
        if verbose == True:
            print('Epoch: {} \tTraining AUPRC score: {:.4f} \tTest AUPRC score: {:.4f} \tTraining Loss: {:.4f} \tTest Loss: {:.4f}'.format(
                epoch, AUPRC_train, AUPRC_test, train_loss, test_loss))
    
    

        # early stop the model if the test loss is not improving
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print('Early stopping the training')
            break

  
    # return the scores at each epoch + the AUPRC, precision and recall
    return AUPRC_train_scores, AUPRC_test_scores, F1_precision_recall_test_scores





class Param_Search():

    """Performs the hyper parameters tuning by using an optimiser among 
    TPE (Tree-structured Parzen Estimator), Bayesian Optimiser and 
    random sampler.  
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    criterion : loss function for training the model.
    num_epochs (int): number of epochs.
    study_name (str): name of the Optuna study object.
    sampler (str): type of optimiser to use. Possible values are
        'BO' (bayesian optimiser), TPE (tree-parzen estimatore) and
        'random' (random sampler).
    n_trial (int): number of trials to perform in the Optuna study.
        Default: 4
    
    Attributes:
    ------------------
    best_model: stores the weights of the common layers of the best performing model.
    
    Returns:
    ------------------
    Prints values of the optimised hyperparameters and saves the parameters of the best model.
    """

    def __init__(self, 
               model,
               train_loader, 
               test_loader,
               criterion,
               num_epochs,
               study_name,
               device,
               sampler='BO',
               n_trials=4
               ):
        self.model_ = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.study_name = study_name
        self.device = device
        self.sampler = sampler
        self.n_trials = n_trials
        self.best_model = None


        # generate the model
        # FFNN needs also the shape of the input as parameter, 
        # while CNN don't
        self.model_name = model.__name__
        
        if sampler == 'BO':
            self.sampler = BoTorchSampler()
        elif sampler == 'TPE':
            self.sampler = TPESampler()
        elif sampler == 'random':
            self.sampler = RandomSampler()
    

    def objective(self, trial):
        """Defines the objective to be optimised (AUPRC test score) and saves
        each final model.
        """

        if self.model_name.startswith('FFNN'):
            input_size = get_input_size(self.train_loader)
            self.model = self.model_(trial, in_features=input_size, device=self.device) #.to(self.device)
        else:
            self.model = self.model_(trial, device=self.device) #.to(self.device)

        # generate the possible optimizers

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Adamax"])

        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)

        if optimizer_name == 'Nadam':
            optimizer = getattr(timm.optim , optimizer_name)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = getattr(torch.optim , optimizer_name)(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # convert model data type to double
        self.model = self.model.double()
        
        early_stopping = EarlyStopping(patience=5, verbose=True)

        # Define the training and testing phases
        for epoch in tqdm(range(1, self.num_epochs + 1), desc='Epochs'):
            train_loss = 0.0
            test_loss = 0.0
            AUPRC_test = 0.0

            # set the model in training modality
            self.model.to(self.device)
            self.model.train()
            for data, target in self.train_loader:
                
                #target = target.reshape(-1,1)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data.double().to(self.device))
                # calculate the batch loss as a sum of the single losses
                try:
                    loss = self.criterion.double().to(self.device)(output.float().to(self.device), target.squeeze().to(self.device)) 
                except:
                    loss = self.criterion.float().to(self.device)(output.float().to(self.device), target.squeeze().to(self.device)) 
                # backward pass: compute gradient of the loss wrt model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()
            
            # set the model in testing modality
            self.model.eval()
            for data, target in self.test_loader:  
                
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data.double().to(self.device))
                # calculate the batch loss as a sum of the single losses
                try:
                    loss = self.criterion.double().to(self.device)(output.float().to(self.device), target.squeeze().to(self.device)) 
                except:
                    loss = self.criterion.float().to(self.device)(output.float().to(self.device), target.squeeze().to(self.device)) 
                # update test loss 
                test_loss += loss.item()
                # calculate AUPRC test score as weighted sum of the single AUPRC scores
                AUPRC_test += AUPRC(output,target)
                

              # calculate epoch score by dividing by the number of observations
            AUPRC_test /= (len(self.test_loader))
        
            # pass the score of the epoch to the study to monitor the intermediate objective values
            trial.report(AUPRC_test, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            early_stopping(test_loss)
            if early_stopping.early_stop:
                print('Early stopping the training')
                break

        # save the final model named with the number of the trial 
        with open("{}{}.pickle".format(self.study_name, trial.number), "wb") as fout:
            pickle.dump(self.model, fout)
        
        # return AUPRC score to the study
        return AUPRC_test



    def run_trial(self):
        """Runs Optuna study and stores the best model in class attribute 'best_model'."""
        
        # create a new study or load a pre-existing study. use sqlite backend to store the study.
        study = optuna.create_study(study_name=self.study_name, direction="maximize",
                                    pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2), 
                                    storage='sqlite:///SA_optuna_tuning.db', load_if_exists=True,
                                    sampler=self.sampler)
        
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        
        study.optimize(self.objective, n_trials=self.n_trials)
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
            
        # store the best model found in the class
        with open("{}{}.pickle".format(self.study_name, study.best_trial.number), "rb") as fin:
            best_model = pickle.load(fin)

        self.best_model = best_model

    
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.best_params = trial.params
        self.trial_value = trial.value

    
    def save_best_model(self, path):
        """Saves the weights of the common layers of the best performing model.
        
        Parameters:
        ------------------
        path: path where the model will be stored.
        
        Returns:
        ------------------
        Weights of the common layers of the best model.
        """
        
        # retrieve the weights of the best model
        model_param = self.best_model.state_dict()
        
        # save only the weights of the common layers
        for key,value in model_param.copy().items():
            if re.findall('last', key):
                del model_param[str(key)]

        basepath = 'models' 
        path = os.path.join(basepath, path)

        torch.save(model_param, path)

        return model_param


def dd():
    return defaultdict(list)


class Kfold_CV():
    """Performs repeated holdout.
    Used for comparing different types of models and with and without augmentation"""
    
    def __init__(self):
        
        self.scores_dict = defaultdict(dd)
        self.model_ = []
        self.optimizer = []
    
    
    def build_dataloader_forCV(self, X, y, sequence, batch_size, training=True,
                        augmentation=True, type_augm_genfeatures='smote', n_neighbors=5):

        if isinstance(X, list):
            for x_ in X:
                x_.reset_index(drop=True, inplace=True)
            for y_ in y:
                y_.reset_index(drop=True, inplace=True)
            
            X = pd.concat([x_ for x_ in X])
            y = pd.concat([y_ for y_ in y])
            
            
        else:
            X.reset_index(drop=True, inplace=True), y.reset_index(drop=True, inplace=True)

        if training and augmentation:
            # augment data
            X, y = data_augmentation(X, y, sequence=sequence, type_augm_genfeatures=type_augm_genfeatures,
                threshold=0.1)

        # build wrapper
        wrap = Dataset_Wrap(X, y, sequence=sequence)
        
        if training:    
            # create balanced dataloader for training set
            return DataLoader(dataset = wrap, 
                       batch_sampler = BalancePos_BatchSampler(wrap, batch_size= batch_size))
            #return DataLoader(dataset = wrap, batch_size= batch_size, shuffle=True)
        else:
            # create dataloader for test set
            return DataLoader(dataset = wrap, batch_size= batch_size*2, shuffle=True)
        
        
        
    def hyper_tuning(self, train_loader, test_loader, num_epochs,
                     study_name, hp_model_path, device, sampler):
            
            param_search = Param_Search(model=self.model_, train_loader=train_loader, 
                                        test_loader=test_loader, criterion=self.criterion, 
                                        num_epochs=num_epochs, device=device, sampler=sampler,
                                        n_trials=3, study_name=study_name)

            param_search.run_trial()
            # retrieve the best trained parameters
            best_params = param_search.best_params
            
            # retrieve the best model
            self.model_ = param_search.best_model
            # reset weights
            self.model_.apply(weight_reset)

            lr = best_params['lr']
            weight_decay = best_params['weight_decay']
            optimizer_name = best_params['optimizer']

            if optimizer_name == 'Nadam':
                self.optimizer = getattr(timm.optim , optimizer_name)(self.model_.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                self.optimizer = getattr(torch.optim , optimizer_name)(self.model_.parameters(), lr=lr, weight_decay=weight_decay)


                
            # save the params of the best hp tuning model (for loading in embracenet)
            self.hp_score.append(param_search.trial_value)
            if param_search.trial_value == max(self.hp_score):
                param_search.save_best_model(hp_model_path) 
    
    
    def model_testing(self, train_loader, test_loader, num_epochs,
                      test_model_path, device, n_of_iterarion):
        
        AUPRC_train, AUPRC_test, other_scores = fit(model=self.model_, train_loader=train_loader, #OTHER_SCORES
                                    test_loader=test_loader, criterion=self.criterion,
                                    device=device, optimizer=self.optimizer, num_epochs=num_epochs, 
                                    patience=5, verbose=False)
            
        self.scores_dict[f'iteration_n_{n_of_iterarion}'][f'AUPRC_train'] = AUPRC_train
        self.scores_dict[f'iteration_n_{n_of_iterarion}'][f'AUPRC_test'] = AUPRC_test
        self.scores_dict[f'iteration_n_{n_of_iterarion}'][f'F1_precision_recall'] = other_scores
            
        print(f'AUPRC test score: {AUPRC_test[-1]}\n\n')
        #print(f'F1: {other_scores[0]}, Precision: {other_scores[1]}, Recall: {other_scores[2]}')
            
        # save the params of the best testing model (for loading in embracenet)
        self.avg_score.append(AUPRC_test[-1])
        if AUPRC_test[-1] == max(self.avg_score):
                save_best_model(self.model_, test_model_path) 
    
    
    
    def __call__(self,
            build_dataloader_pipeline, 
            cell_line,
            device,
            sequence=False, 
            model=None,
            augmentation=False,
            type_augm_genfeatures='smote',
            random_state=321,
            n_folds=4,
            num_epochs=50, 
            batch_size=100,
            study_name=None,
            sampler='BO',
            hp_model_path=None, #ex: FFNN/best_model_FFNN_hp
            test_model_path=None # ex: FNN/best_model_FFNN_test
            ):
        
        self.n_folds = n_folds
    
        self.avg_score = []
        self.hp_score = []
        
        data_class = build_dataloader_pipeline.data_class
        kf, X, y = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=sequence,
                                                      n_folds=n_folds, random_state=random_state)
        
        w_pos,w_neg = get_loss_weights_from_labels(y)
        self.criterion=nn.CrossEntropyLoss(weight=torch.tensor([w_pos,w_neg]))
            
        
        i=1
        for train_index, test_index in kf.split(X):
            
            study_name = study_name + '_' + str(i)

            print(f'>>> ITERATION N. {i}')
            i+=1
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=1/self.n_folds,
                                                              random_state=random_state, shuffle=True) 
            self.model_ = model
            
            print('\n===============> HYPERPARAMETERS TUNING')
            
            train_loader = self.build_dataloader_forCV(X_train, y_train, sequence=sequence, 
                                               batch_size=batch_size, training=True,
                                               augmentation=augmentation, type_augm_genfeatures=type_augm_genfeatures)
            test_loader = self.build_dataloader_forCV(X_val, y_val, sequence=sequence, 
                                               batch_size=batch_size, training=False,
                                               augmentation=False)
            
            self.hyper_tuning(train_loader, test_loader, num_epochs,
                              study_name, hp_model_path, device, sampler)
            
            
            print('\n===============> MODEL TESTING')
            
            train_loader = self.build_dataloader_forCV([X_train, X_val], [y_train, y_val], sequence=sequence, 
                                               batch_size=batch_size, training=True,
                                               augmentation=augmentation, type_augm_genfeatures=type_augm_genfeatures)
            test_loader = self.build_dataloader_forCV(X_test, y_test, sequence=sequence, 
                                               batch_size=batch_size, training=False,
                                               augmentation=False)
            
            self.model_testing(train_loader, test_loader, num_epochs,
                               test_model_path, device, i)
                
        
        print(f'\n{n_folds}-FOLD CROSS-VALIDATION AUPRC TEST SCORE: {np.round(sum(self.avg_score)/n_folds, 5)}')
            


def plot_results(self):
        
        for i in self.n_folds:
            print(f'ITERATION N. {i}')
            plot_F1_scores(self.scores_dict[f'trial_n_{i}'][f'AUPRC_train'],
                              self.scores_dict[f'trial_n_{i}'][f'AUPRC_test'])
            
            
            #plot_other_scores(self.scores_dict[f'iteration_n_{n_of_iterarion}'][f'AUPRC_precision_recall'])
    
    


    
