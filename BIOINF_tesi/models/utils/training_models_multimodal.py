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
    get_input_size, 
    selection_probabilities)
from BIOINF_tesi.data_pipe.dataprepare import Data_Prepare, Dataset_Wrap, BalancePos_BatchSampler
from BIOINF_tesi.data_pipe.utils import data_augmentation, get_imbalance, data_rebalancing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CELL_LINES = ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']
TASKS = ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 
            'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_rest']



def fit_multimodal(model, 
        train_loader, 
        test_loader, 
        device,
        cell_line,
        task,
        optimizer=None, 
        num_epochs=100,
        patience=4,
        delta=0,
        verbose=True,
        checkpoint_path=None): 
    
    """Performs the training of the model. Store the trained model or load it if 
    the model has been already trained.
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    device: 'cpu' or 'cuda'.
    cell_line: cell_line used among ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7'].
    task: task used among ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 
            'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_rest'].
    optimizer (torch.optim): optimization algorithm for training the model. 
    num_epochs (int): number of epochs.
    patience (int): number of epochs in which the test error is not anymore decreasing
        before stopping the training.
        Default:4
    delta (int): minimum decrease in the test error to continue with the training.
        Default:0
    verbose (bool): prints the training error, test error, F1 training score, F1 test score 
        at each epoch.
        Default: True
    checkpoint_path (str): path where the final model and scores will be stored.
    
    
    Returns:
    ------------------
    Lists of AUPRC_train_scores, AUPRC_test_scores, F1_precision_recall_test_scores at each epoch.
    Prints AUPRC training score, AUPRC test score training error, test error at each epoch.
    """

    if cell_line not in CELL_LINES:
            raise ValueError(
            f"Argument 'cell_line' has an incorrect value: use one among {CELL_LINES}")

    if task not in TASKS:
            raise ValueError(
                f"Argument 'task' has an incorrect value: use one among {TASKS} ")


    # if the model has been already trained, load the optimised weights into the model,
    #and the scores.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        AUPRC_train_scores = checkpoint['AUPRC_train_scores']
        AUPRC_test_scores = checkpoint['AUPRC_test_scores']
        F1_precision_recall_test_scores = checkpoint['F1_precision_recall_test_scores']
    
    else:

        # keep track of epoch losses 
        AUPRC_train_scores = []
        AUPRC_test_scores = []
        F1_precision_recall_test_scores = []
    
        # save model name 
        try:
            model_name = model.__name__
        except:
            model_name = type(model).__name__ 
        # convert model data type to double and load on device
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

            for load1, load2 in zip(train_loader['FFNN'], train_loader['CNN']):
                x_1, target = load1
                x_2, _ = load2
                assert(len(x_1)==len(x_2))
                assert(torch.eq(target, _).all())

                # calculates the batch weights of the classes and define the weighted
                #loss function
                w_pos,w_neg = get_loss_weights_from_labels(target)
                criterion=nn.CrossEntropyLoss(weight=torch.tensor([w_neg, w_pos]))

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model.
                if model_name == 'EmbraceNetMultimodal':
                    output = model([x_1.double(), x_2.double()], is_training=True)
                else:
                    output = model([x_1.double(), x_2.double()])
                # calculate the batch loss.
                try:
                    loss = criterion.double().to(device)(output.float(), target.squeeze()) 
                except:
                    loss = criterion.float().to(device)(output.float(), target.squeeze()) 
                # backward pass: compute gradient of the loss wrt model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()
                # calculate AUPRC training score
                auprc = AUPRC(output,target)
                AUPRC_train += auprc


            # set the model in testing modality
            model.eval()
            for load1, load2 in zip(test_loader['FFNN'],
                                          test_loader['CNN']):
                x_1, target = load1
                x_2, _, = load2
                assert(len(x_1)==len(x_2))
                assert(torch.eq(target, _).all())

                # calculates the batch weights of the classes and define the weighted
                #loss function
                w_pos,w_neg = get_loss_weights_from_labels(target)
                criterion=nn.CrossEntropyLoss(weight=torch.tensor([w_neg, w_pos]))

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model([x_1.double(), x_2.double()])
                # calculate the batch loss
                try:
                    loss = criterion.double().to(device)(output.float(), target.squeeze()) 
                except:
                    loss = criterion.float().to(device)(output.float(), target.squeeze())
                # update test loss
                test_loss += loss.item()
                # calculate AUPRC test score and F1_precision_recall 
                auprc = AUPRC(output,target)
                AUPRC_test += auprc
                F1_precision_recall_test += F1_precision_recall(output,target)

            # calculate epoch scores by dividing by the number of observations
            AUPRC_train /= (len(train_loader['FFNN']))
            AUPRC_test /= (len(test_loader['FFNN']))
            F1_precision_recall_test /= (len(test_loader['FFNN']))

            # store epoch scores
            AUPRC_train_scores.append(AUPRC_train) 
            AUPRC_test_scores.append(AUPRC_test)
            F1_precision_recall_test_scores.append(F1_precision_recall_test)

            # print training/test statistics 
            if verbose == True:
                print('Epoch: {} \tTraining AUPRC score: {:.4f} \tTest AUPRC score: {:.4f} \tTraining Loss: {:.4f} \tTest Loss: {:.4f}'.format(
                    epoch, AUPRC_train, AUPRC_test, train_loss, test_loss))


            # early stop the model if the AUPRC is not improving
            early_stopping(AUPRC_test)
            if early_stopping.early_stop:
                print('Early stopping the training')
                break

        # if a checkpoint path has been defined and when the model has finished being trained, 
        #save the optimised weights and the scores.
        if checkpoint_path:
            torch.save({
                'model_state_dict': model.state_dict(),
                'AUPRC_train_scores': AUPRC_train_scores,
                'AUPRC_test_scores': AUPRC_test_scores,
                'F1_precision_recall_test_scores': F1_precision_recall_test_scores
            }, checkpoint_path)

    return AUPRC_train_scores, AUPRC_test_scores, F1_precision_recall_test_scores





class Param_Search_Multimodal():

    """Performs the hyper parameters tuning by using an optimiser among 
    TPE (Tree-structured Parzen Estimator), Bayesian Optimiser and 
    random sampler.  
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    num_epochs (int): number of epochs.
    study_name (str): name of the Optuna study object.
    device (str): cpu or cuda.
    sampler (str): type of optimiser to use. Possible values are
        'BO' (bayesian optimiser), TPE (tree-parzen estimatore) and
        'random' (random sampler).
    n_trial (int): number of trials to perform in the Optuna study.
        Default: 3
    storage (str): sqlite backend name.
        Default: 'BIOINF_optuna_tuning.db'
    
    Attributes:
    ------------------
    best_model: stores the weights of the common layers of the best performing model.
    best_params: stores the parameters of the best performing model. 
    
    Returns:
    ------------------
    Prints values of the optimised hyperparameters and saves the parameters of the best model.
    """

    def __init__(self, 
               model,
               train_loader, 
               test_loader,
               num_epochs,
               study_name,
               device,
               cell_line,
               task,
               sampler='TPE',
               n_trials=3,
               storage = 'BIOINF_optuna_tuning.db' 
               ):
        self.model_ = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.study_name = study_name
        self.device = device
        self.cell_line = cell_line
        self.task = task
        self.n_trials = n_trials
        self.storage = storage


        if cell_line not in CELL_LINES:
            raise ValueError(
            f"Argument 'cell_line' has an incorrect value: use one among {CELL_LINES}")

        if task not in TASKS:
            raise ValueError(
                f"Argument 'task' has an incorrect value: use one among {TASKS} ")

        # save the model name
        self.model_name = model.__name__
        # define sampler algorithm
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

        # generate the model 
        in_features_FFNN = get_input_size(self.train_loader['FFNN'])
        self.model = self.model_(trial, cell_line=self.cell_line, task=self.task,
            device=self.device, in_features_FFNN=in_features_FFNN) 

        # generate the possible optimizers, learning rate and weight decay
        optimizer_name = trial.suggest_categorical("optimizer", ["Nadam", "Adam", "RMSprop"])
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)

        if optimizer_name == 'Nadam':
            optimizer = getattr(timm.optim , optimizer_name)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = getattr(torch.optim , optimizer_name)(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # convert model data type to double and load it into device
        self.model = self.model.double().to(self.device)
        # define early stopping
        early_stopping = EarlyStopping(patience=4, verbose=True)


        for epoch in tqdm(range(1, self.num_epochs + 1), desc='Epochs'):
            train_loss = 0.0
            test_loss = 0.0
            AUPRC_test = 0.0
        
            # set the model in training modality
            self.model.train()
            i=0
            for load1, load2 in zip(self.train_loader['FFNN'],
                                          self.train_loader['CNN']):
                x_1, target = load1
                x_2, _, = load2
                assert(len(x_1)==len(x_2))
                assert(torch.eq(target, _).all())

                # calculates the batch weights of the classes and define the weighted
                #loss function
                w_pos,w_neg = get_loss_weights_from_labels(target)
                self.criterion=nn.CrossEntropyLoss(weight=torch.tensor([w_neg, w_pos]))

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                if self.model_name == 'EmbraceNetMultimodal':
                    output = self.model([x_1.double(), x_2.double()], is_training=True)
                else:
                    output = self.model([x_1.double(), x_2.double()])
                # calculate the batch loss
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
            for load1, load2 in zip(self.test_loader['FFNN'],
                                          self.test_loader['CNN']):
                x_1, target = load1
                x_2, _, = load2
                assert(len(x_1)==len(x_2))
                assert(torch.eq(target, _).all())

                # calculates the batch weights of the classes and define the weighted
                #loss function
                w_pos,w_neg = get_loss_weights_from_labels(target)
                self.criterion=nn.CrossEntropyLoss(weight=torch.tensor([w_neg, w_pos]))

                output = self.model([x_1.double(), x_2.double()]) 
                # calculate the batch loss
                try:
                    loss = self.criterion.double().to(self.device)(output.float().to(self.device), target.squeeze().to(self.device)) 
                except:
                    loss = self.criterion.float().to(self.device)(output.float().to(self.device), target.squeeze().to(self.device)) 
                # update test loss 
                test_loss += loss.item()
                AUPRC_test += AUPRC(output,target)
                

            # calculate epoch score by dividing by the number of observations
            AUPRC_test /= (len(self.test_loader['FFNN']))
        
            # pass the score of the epoch to the study to monitor the intermediate objective values
            trial.report(AUPRC_test, epoch)
            # stop the optimisation if the trial is not promising
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # early stop the model if the AUPRC is not improving
            early_stopping(AUPRC_test)
            if early_stopping.early_stop:
                print('Early stopping the training')
                break

        # save the final model named with the number of the trial 
        torch.save(self.model, f'{self.study_name}{trial.number}.pt')
        
        # return AUPRC score to the study
        return AUPRC_test



    def run_trial(self):
        """Runs Optuna study and stores the best model in class attribute 'best_model'."""
        
        # create a new study or load a pre-existing study. use sqlite backend to store the study.
        study = optuna.create_study(study_name=self.study_name, direction="maximize",
                                    pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2), 
                                    storage=f'sqlite:///{self.storage}', load_if_exists=True,
                                    sampler=self.sampler)
        
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
            
        # if the number of already completed trials is lower than the total number of trials passed as
        #argument, perform the remaining trials 
        if len(complete_trials)<self.n_trials:
            # set the number of trials to be performed equal to the number of missing trials
            self.n_trials -= len(complete_trials)
            study.optimize(self.objective, n_trials=self.n_trials)
            pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
            
            
        # load the best model found in the class
        best_model = torch.load(f'{self.study_name}{study.best_trial.number}.pt', torch.device(self.device)) 
        # save the best model
        self.best_model = best_model

    
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        # save parameters of best trial
        trial = study.best_trial
        self.best_params = trial.params

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))



def dd():
    return defaultdict(list)

def path_augmentation(x):
    return '_augmentation' if x==True else ''



class Kfold_CV_Multimodal():
    """Performs k-folds cross-validation. At each iteration performs
    hyperparameter tuning, then train and test the model with the optimised
    hyperparameters found.
    Prints the average cross-validation AUPRC.
    """
    
    def __init__(self):
        
        self.scores_dict = defaultdict(dd)
        self.scores_dict['final_test_AUPRC_scores'] = []
        self.scores_dict['final_train_AUPRC_scores'] = []

        self.model_ = []
        self.optimizer = []
        self.best_params = defaultdict(dict)
    
    
    def build_dataloader_forCV(self, X, y, sequence, batch_size=100, training=True,
                        augmentation=False):

        """Builds DataLoader object for training or testing.
    
        Parameters:
        ------------------
        X (pd.DataFrame): data.
        y (pd.Seres): labels.
        sequence (bool): whether the data are genomic sequence or not.
        batch_size (int): size of the batch.
            Default: 100
        training (bool): whether the DataLoader is for training or testing.
            Default: True
        augmentation (bool): whether or not to augment data.
            Default: False
        
        Returns:
        ------------------
        Pytorch DataLoader.
        """

        # fix data format
        if isinstance(X, list):
            for x_ in X:
                x_.reset_index(drop=True, inplace=True)
            for y_ in y:
                y_.reset_index(drop=True, inplace=True)
            
            X = pd.concat([x_ for x_ in X])
            y = pd.concat([y_ for y_ in y])
        else:
            X.reset_index(drop=True, inplace=True), y.reset_index(drop=True, inplace=True)


        if training:
            # if training=True and augmentation=True, augment and rebalance the dataset if needed
            if augmentation:
                X, y = data_augmentation(X, y, sequence=sequence, rebalance_threshold=self.rebalance_threshold)
            # if training=True and the dataset is imbalanced, perform rebalancing
            elif not augmentation and get_imbalance(y) < self.rebalance_threshold:
                X, y = data_rebalancing(X, y, sequence=sequence, rebalance_threshold=self.rebalance_threshold)

        # build wrapper
        wrap = Dataset_Wrap(X, y, sequence=sequence)
        if training:    
            # create dataloader with balanced batches for training set
            return DataLoader(dataset = wrap, 
                       batch_sampler = BalancePos_BatchSampler(wrap, batch_size= batch_size))
            return DataLoader(dataset = wrap, batch_size= batch_size, shuffle=True)
        else:
            # create dataloader for test set
            return DataLoader(dataset = wrap, batch_size= batch_size*2, shuffle=True,
                            generator=torch.Generator().manual_seed(self.random_state+30)) #########   
        
        
    def hyper_tuning(self, train_loader, test_loader, num_epochs, cell_line, task,
                     study_name, device, sampler):
            
            """Perform hyperparameter tuning through Optuna framework.
    
            Parameters:
            ------------------
            train_loader (DataLoader): DataLoader object for training. 
            test_loader (DataLoader): DataLoader object for testing. 
            num_epochs (int): number of epochs.
            cell_line (str): cell line used among ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7'].
            task (str): task used among ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 
                'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_rest'].
            study_name (str): name of the Optuna study object.
            device (str): cpu or cuda.
            sampler (str): type of optimiser to use. Possible values are
                'BO' (bayesian optimiser), TPE (tree-parzen estimatore) and
                'random' (random sampler).
            """

            param_search = Param_Search_Multimodal(model=self.model_, train_loader=train_loader, 
                                        test_loader=test_loader, 
                                        num_epochs=num_epochs, cell_line=cell_line, task=task, 
                                        device=device, sampler=sampler,
                                        n_trials=3, study_name=study_name)

            param_search.run_trial()
            # retrieve the best trained parameters
            best_params = param_search.best_params
            # retrieve the best model and its hyperparameters
            self.model_ = param_search.best_model
            self.best_params[self.i] = best_params
            # reset weights of the best model to use it for testing
            self.model_.apply(weight_reset)

            # set optimiser with optimised hyperparameters
            lr = best_params['lr']
            weight_decay = best_params['weight_decay']
            optimizer_name = best_params['optimizer']
            if optimizer_name == 'Nadam':
                self.optimizer = getattr(timm.optim , optimizer_name)(self.model_.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                self.optimizer = getattr(torch.optim , optimizer_name)(self.model_.parameters(), lr=lr, weight_decay=weight_decay)

    
    
    def model_testing(self, train_loader, test_loader, num_epochs,
                      test_model_path, device, cell_line, task, checkpoint_path=None):

        """Perform training and testing of the model using the previously optimised hyperparameters.
    
            Parameters:
            ------------------
            train_loader (DataLoader): DataLoader object for training. 
            test_loader (DataLoader): DataLoader object for testing. 
            num_epochs (int): number of epochs.
            test_model_path (str): path where to save only the final best model of the K-folds CV.
            cell_line (str): cell line used among ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7'].
            task (str): task used among ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 
                'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_rest'].
            checkpoint_path (str): path where to save all trained models.
                Default: None.
            """

        # train and test model with previously optimised hyperparameters
        AUPRC_train, AUPRC_test, other_scores = fit_multimodal(model=self.model_, 
                                    train_loader=train_loader, 
                                    test_loader=test_loader, 
                                    device=device, cell_line=cell_line, task=task,
                                    optimizer=self.optimizer, num_epochs=num_epochs, 
                                    patience=4, verbose=False, 
                                    checkpoint_path=f'{checkpoint_path}.pt')
        
        # store scores of each epoch
        self.scores_dict[f'iteration_n_{self.i}'][f'AUPRC_train'] = AUPRC_train
        self.scores_dict[f'iteration_n_{self.i}'][f'AUPRC_test'] = AUPRC_test
        self.scores_dict[f'iteration_n_{self.i}'][f'F1_precision_recall'] = other_scores
            
        # retrieve and save final AUPRC score of the model
        final_test_AUPRC_score = AUPRC_test[-1]
        self.scores_dict['final_test_AUPRC_scores'].append(final_test_AUPRC_score)
        final_train_AUPRC_score = AUPRC_train[-1]
        self.scores_dict['final_train_AUPRC_scores'].append(final_train_AUPRC_score)

        print(f'AUPRC test score: {final_test_AUPRC_score}\n\n')
        
        # save the weights, hyperparameters and architecture of the final best trained model of
        #the K-folds CV.
        self.avg_score.append(final_test_AUPRC_score)
        if final_test_AUPRC_score == max(self.avg_score):
                torch.save({
                        'model_state_dict': self.model_.state_dict(),
                        'model_params': self.best_params[self.i]
                    }, f'models/{test_model_path}{path_augmentation(self.augmentation)}.pt')
    
    
    def __call__(self,
            build_dataloader_pipeline, 
            cell_line,
            device,
            task=None,
            model=None,
            augmentation=False,
            rebalance_threshold=0.1,
            random_state=789,
            n_folds=3,
            num_epochs=100, 
            batch_size=100,
            study_name=None,
            sampler='TPE',
            test_model_path=None 
            ):
        
        """
            Attributes:
            ------------------
            scores_dict: dictionary containing all the relevant scores of the models.

            Parameters:
            ------------------
            build_dataloader_pipeline: Build_DataLoader_Pipeline object with pre-processed data.
            cell_line (str): cell line used among ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7'].
            device (str): cpu or cuda.
            task (str): task used among ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 'active_E_vs_active_P', 
                'inactive_E_vs_inactive_P', 'active_EP_vs_inactive_rest'].
                Default: None
            model (torch.nn.Module): neural network model.
                Default: None
            augmentation (bool): whether to augment or not the dataset.
                Default: False
            rebalance_threshold (float): minimum desired imbalance between classes.
                Default: 0.1
            random_state (int): initial seed.
                Default: 789
            n_folds (int): number of folds for the cross-validation.
                Default: 3
            num_epochs (int): number of epochs.
                Default: 100
            batch_size (int): size of the training batches. Test batches have size batch_size*2
                Default: 100
            study_name (str): name of the Optuna study object.
                Default: None
            sampler (str): type of optimiser to use. Possible values are
                'BO' (bayesian optimiser), TPE (tree-parzen estimatore) and
                'random' (random sampler).
            test_model_path (str): path where to save only the final best model of the K-folds CV.
                Default: None
        """

        self.n_folds = n_folds
        self.augmentation = augmentation
        self.rebalance_threshold = rebalance_threshold
        self.random_state = random_state
        self.device = device

        self.avg_score = []
        self.hp_score = []

        if cell_line not in CELL_LINES:
            raise ValueError(
            f"Argument 'cell_line' has an incorrect value: use one among {CELL_LINES}")

        if task not in TASKS:
            raise ValueError(
                f"Argument 'task' has an incorrect value: use one among {TASKS} ")


        # extract epigenomic features and genomic sequence data for the specified cell line and task.
        # also extract indexes for performing cross-validation.
        data_class = build_dataloader_pipeline.data_class 
        kf, X_1, y = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=False, 
                                                      n_folds=n_folds, random_state=self.random_state)
        _, X_2, _ = data_class.return_index_data_for_cv(cell_line=cell_line, sequence=True, 
                                                      n_folds=n_folds, random_state=self.random_state)
    
        # start CV iterations
        for i, (train_index, test_index) in enumerate(kf.split(X_1)):
            self.i = i+1
            
            study_name = f'{study_name}_{str(self.i)}{path_augmentation(self.augmentation)}'

            print(f'>>> ITERATION N. {self.i}')
            
            # prepare training set, validation set and test set.
            X_train_1, X_test_1 = X_1.iloc[train_index], X_1.iloc[test_index]
            X_train_2, X_test_2 = X_2.iloc[train_index], X_2.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train_1, X_val_1, _, _ = train_test_split(X_train_1, y_train,
                                                              test_size=1/self.n_folds,
                                                              random_state=self.random_state, shuffle=True) 
            X_train_2, X_val_2, y_train, y_val = train_test_split(X_train_2, y_train,
                                                              test_size=1/self.n_folds,
                                                              random_state=self.random_state, shuffle=True) 
            # define model
            self.model_ = model
            
            print('\n===============> HYPERPARAMETERS TUNING')
            
            # prepare DataLoader for training and testing of hyperparameter tuning phase.            
            train_loader = defaultdict(dict)

            train_loader['FFNN'] = self.build_dataloader_forCV(X_train_1, y_train,
                                               batch_size=batch_size, training=True, sequence=False,
                                               augmentation=self.augmentation)
            train_loader['CNN'] = self.build_dataloader_forCV(X_train_2, y_train,
                                               batch_size=batch_size, training=True, sequence=True,
                                               augmentation=self.augmentation)
            test_loader = defaultdict(dict)
            test_loader['FFNN'] = self.build_dataloader_forCV(X_val_1, y_val, sequence=False,
                                               batch_size=batch_size, training=False,
                                               augmentation=False) 
            test_loader['CNN'] = self.build_dataloader_forCV(X_val_2, y_val, sequence=True,
                                               batch_size=batch_size, training=False,
                                               augmentation=False) 

            # perform hyperparameter tuning.
            self.hyper_tuning(train_loader, test_loader, num_epochs, cell_line, task,
                              study_name, device, sampler)
            
            
            print('\n===============> MODEL TESTING')
            
            # prepare DataLoader for training and testing of final model.            
            train_loader = defaultdict(dict)
            train_loader['FFNN'] = self.build_dataloader_forCV([X_train_1, X_val_1], [y_train, y_val], sequence=False, 
                                               batch_size=batch_size, training=True,
                                               augmentation=self.augmentation)
            train_loader['CNN'] = self.build_dataloader_forCV([X_train_2, X_val_2], [y_train, y_val], sequence=True, 
                                               batch_size=batch_size, training=True,
                                               augmentation=self.augmentation)

            test_loader = defaultdict(dict)
            test_loader['FFNN'] = self.build_dataloader_forCV(X_test_1, y_test, sequence=False, 
                                               batch_size=batch_size, training=False,
                                               augmentation=False)
            test_loader['CNN'] = self.build_dataloader_forCV(X_test_2, y_test, sequence=True, 
                                               batch_size=batch_size, training=False,
                                               augmentation=False)

            # perform testing of the final model            
            self.model_testing(train_loader, test_loader, num_epochs,
                               test_model_path, device, cell_line, task,
                               checkpoint_path= f'{cell_line}_{model.__name__}_{task}_{self.i}{path_augmentation(self.augmentation)}') 

        # compute average AUPRC of the CV
        avg_CV_AUPRC = np.round(sum(self.avg_score)/n_folds, 5)
        self.scores_dict['average_CV_AUPRC'] = avg_CV_AUPRC
        print(f'\n{n_folds}-FOLD CROSS-VALIDATION AUPRC TEST SCORE: {avg_CV_AUPRC}')
            