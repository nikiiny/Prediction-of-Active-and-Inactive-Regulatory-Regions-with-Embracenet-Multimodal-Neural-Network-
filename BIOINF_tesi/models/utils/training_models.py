import pandas as pd
import numpy as np
import os
import pickle
from tqdm.auto import tqdm
import sqlite3
from sqlalchemy import create_engine

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import botorch
from optuna.integration import BoTorchSampler

from .utils import EarlyStopping, accuracy, F1


def fit(model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer=None, 
        num_epochs=50,
        filename_path=None, 
        patience=3,
        sequence=False,
        delta=0,
        verbose=True): 
    
  """Performs the training of the model. It implements also early stopping
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    criterion: loss function for training the model.
    optimizer (torch.optim): optimization algorithm for training the model. 
    num_epochs (int): number of epochs.
    filename_path (str): where the weights of the model at each epoch will be stored. 
        Indicate only the name of the folder.
    patience (int): number of epochs in which the test error is not anymore decreasing
        before stopping the training.
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

   # gdrive_path = '/content/gdrive/MyDrive/Thesis_BIOINF' ###
    basepath = 'exp'
   #basepath = gdrive_path + basepath ###



    # keep track of epoch losses 
    f1_train_scores = []
    f1_test_scores = []

    # convert model data type to double
    model = model.double()

    # define early stopping
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    
    
    for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
        train_loss = 0.0
        test_loss = 0.0
        
        f1_train = 0.0
        f1_test = 0.0

    
        # if there is already a trained model stored for a specific epoch, load the model
        #and don't retrain the model
        if os.path.exists( os.path.join(basepath, filename_path + '_' + str(epoch) + '.pt') ):
            
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            f1_train = checkpoint['F1_train']
            f1_test = checkpoint['F1_test']
            train_loss = checkpoint['train_loss']
            test_loss = checkpoint['test_loss']
    
        else:
        # set the model in training modality
        model.train()

        for data, target in tqdm(train_loader, desc='Training model'):
        
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.double())
            # calculate the batch loss as the sum of all the losses
            loss = criterion(output, target) 
            # backward pass: compute gradient of the loss wrt model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()
            # calculate F1 training score as a weighted sum of the single F1 scores
            f1_train += F1(output,target)

        
        # set the model in testing modality
        model.eval()
        for data, target in tqdm(test_loader, desc='Testing model'):

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.double())
            # calculate the batch loss as the sum of all the losses
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item()
            # calculate F1 test score as a weighted sum of the single F1 scores
            f1_test += F1(output,target) 
        
    
    
        # save the model weights, epoch, scores and losses at each epoch
        model_param = model.state_dict()
        PATH = os.path.join(basepath, filename_path + '_' + str(epoch) + '.pt')
        torch.save({'epoch': epoch,
                        'model_state_dict': model_param,
                        'F1_train': f1_train,
                        'F1_test': f1_test,
                        'train_loss': train_loss,
                        'test_loss': test_loss},
                       PATH)
    
        # calculate epoch score by dividing by the number of observations
        f1_train /= (len(train_loader))
        f1_test /= (len(test_loader))
        # store epoch score
        f1_train_scores.append(f1_train)    
        f1_test_scores.append(f1_test)
          
        # print training/test statistics 
        if verbose == True:
            print('Epoch: {} \tTraining F1 score: {:.4f} \tTest F1 score: {:.4f} \tTraining Loss: {:.4f} \tTest Loss: {:.4f}'.format(
                epoch, f1_train, f1_test, train_loss, test_loss))
    
    

        # early stop the model if the test loss is not improving
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print('Early stopping the training')
            # reload the previous best model before the test loss started decreasing
            best_checkpoint = torch.load(os.path.join(basepath,filename_path + '_' + '{}'.format(epoch-patience) + '.pt'))
            model.load_state_dict(best_checkpoint['model_state_dict'])
            break

  
    # return the scores at each epoch
    return f1_train_scores, f1_test_scores





class Param_Search():

    """Performs the hyper parameters tuning by using a TPE (Tree-structured Parzen Estimator) 
    algorithm sampler.  
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    criterion : loss function for training the model.
    num_epochs (int): number of epochs.
    study_name (str): name of the Optuna study object.
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
               # network_type,
               model,
               train_loader, 
               test_loader,
               criterion,
               num_epochs,
               study_name,
               input_size,
               n_trials=4
               ):
        # self.network_type = network_type
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.study_name = study_name
        self.sequence = sequence
        self.n_trials = n_trials
        self.best_model = None
    

    def objective(self, trial):
        """Defines the objective to be optimised (F1 test score) and saves
        each final model.
        """

        # generate the model
        # model = FFNN_define_model(trial, in_features_INPUT=self.input_size, classes=2)
        self.model = model

        # generate the possible optimizers
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # convert model data type to double
        model = model.double()

        
        # Define the training and testing phases
        for epoch in tqdm(range(1, self.num_epochs + 1), desc='Epochs'):
            train_loss = 0.0
            test_loss = 0.0
            f1_test = 0.0

            # set the model in training modality
            model.train()
            for data, target in tqdm(self.train_loader, desc='Training Model'):
                

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data.double())
                # calculate the batch loss as a sum of the single losses
                loss = self.criterion(output, target) 
                # backward pass: compute gradient of the loss wrt model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()
            
            # set the model in testing modality
            model.eval()
            for data, target in tqdm(self.test_loader, desc='Testing Model'):  

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data.double())
                # calculate the batch loss as a sum of the single losses
                loss = self.criterion(output, target)
                # update test loss 
                test_loss += loss.item()
                # calculate F1 test score as weighted sum of the single F1 scores
                f1_test += F1(output,target)

              # calculate epoch score by dividing by the number of observations
            f1_test /= (len(self.test_loader))
        
            # pass the score of the epoch to the study to monitor the intermediate objective values
            trial.report(f1_test, epoch)

        # save the final model named with the number of the trial 
        with open("{}{}.pickle".format(self.study_name, trial.number), "wb") as fout:
            pickle.dump(model, fout)
        
        # return F1 score to the study
        return f1_test



    def run_trial(self):
        """Runs Optuna study and stores the best model in class attribute 'best_model'."""
        
        # create a new study or load a pre-existing study. use sqlite backend to store the study.
        study = optuna.create_study(study_name=self.study_name, direction="maximize", 
                                   # storage='sqlite:///SA_optuna_tuning.db', load_if_exists=True,
                                    sampler=BoTorchSampler())
        
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

    def get_best_params(self):
        return self.best_params

    
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

      #  gdrive_path = '/content/gdrive/MyDrive/Thesis_BIOINF' ###
        basepath = 'models' 
       # basepath = grive_path + basepath ###
        path = os.path.join(basepath, path)

        torch.save(model_param, path)

        return model_param
