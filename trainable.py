import os
import numpy as np
import random
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from copy import deepcopy
import sklearn as sk
from sklearn.preprocessing import Binarizer
import time
import pickle
from skorch.dataset import CVSplit
from torch.utils.data import DataLoader, TensorDataset


from ray.tune.stopper import Stopper
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from genome_embeddings import models
from argparse import Namespace
from genome_embeddings import data_viz
from genome_embeddings import evaluate
from genome_embeddings import models
from genome_embeddings import train_test
from genome_embeddings import util
from genome_embeddings import models 

DATA_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_embeddings/data/'

class MemCache:
	train_data = torch.load(DATA_FP+"corrupted_train.pt")
	test_data = torch.load(DATA_FP+"corrupted_test.pt")
	genome_idx_train = torch.load(DATA_FP+"genome_idx_train.pt")
	genome_idx_test = torch.load(DATA_FP+"genome_idx_test.pt")
	
	# To make predictions on (ROC + AUC)
	num_features = int(train_data.shape[1]/2)
	tensor_test_data = torch.tensor([i.numpy() for i in test_data]).float()
	corrupt_test_data = tensor_test_data[:,:num_features]
	target = tensor_test_data[:,num_features:].detach().numpy()
	
		
def binarize(pred_tensor, replacement_threshold):
	"""
	Values below or equal to threshold are replaced by 0, else by 1
	
	Arguments:
	pred_tensor -- numpy array from pred.detach().numpy()
	formerly: torch tensor of probabilities ranging from 0 to 1
	threshold -- threshold at which to replace with 0 vs 1
	
	Returns:
	binary_preds -- list of numpy array of 0's and 1's 
	"""
		
	binary_preds = []
	for i in pred_tensor:
		try:
			pred_arr = i.detach().numpy() #.data.numpy()		
		except AttributeError:
			pred_arr = i
		b = Binarizer(threshold=replacement_threshold).fit_transform(pred_arr.reshape(1, -1))
		binary_preds.extend(b)
	return binary_preds

def f1_score(pred_non_bin, target, replacement_threshold):
	"""
	Calculates F1 score
	
	Arguments:
	pred_non_bin -- torch tensor containing predictions output by model (probability values)
	dataset -- torch tensor containing true values (binary)
	
	Returns:
	avg_f1 -- average F1 score
	"""
	binarized_preds = binarize(pred_non_bin, replacement_threshold)
	
	f1s = []
	for i in range(0,len(binarized_preds)):
		f1 = sk.metrics.f1_score(target.data.numpy()[i], binarized_preds[i])
		f1s.append(f1)
	
	return sum(f1s)/len(f1s)

def get_dataloader(DATA_FP, batch_size, num_features):
	
	# load data from memory (faster than re-loading from scratch every time)
	train_data = MemCache.train_data
	test_data = MemCache.test_data
	genome_idx_train = MemCache.genome_idx_train
	genome_idx_test = MemCache.genome_idx_test
	
	# Create dataloaders for this experimental run
	loaders = util.dataloaders(train_data, test_data, batch_size, batch_size, num_features)	
	
	return loaders


def cv_dataloader(batch_size, num_features, k):

	"""
	Creates training dataloader w/ k-folds for cross validation
	Note: only creates 1 split regardless of # k -- improves training speed (method of skorch CVSplit)
	
	Arguments:
	train_data (tensor) -- training data 
		Each row has the corrupted version of a genome + the uncorrupted genome concatenated together
	batch_size (int) -- batch size for training set
	num_features (int) -- number of features / genes
	k (int) -- number of folds
	
	Returns:
	train_dl (DataLoader) -- X and y training Datasets
	"""
	# load data from memory (faster than re-loading from scratch every time)
	train_data = MemCache.train_data
	#genome_idx_train = MemCache.genome_idx_train
	
	# split X and y
	X = train_data[:,:num_features] # corrupted genomes in first half of matrix columns
	y = train_data[:,num_features:] # uncorrupted in second half of matrix columns
	
	# Create dataloader with folds 
#	train_ds = TensorDataset(X, y)
#	splitter = CVSplit(cv=k)
#	train_dl = splitter(train_ds)

	# Create random split for 1-time k-fold validation
	idx_genomes = [i for i in range(len(X))]
	num_cv = int(len(idx_genomes) / k )
	num_train = len(idx_genomes) - num_cv
	cv_idx = np.random.choice(idx_genomes, num_cv, replace=False)
	train_idx = list(set(idx_genomes) - set(cv_idx))	
	X_train = X[train_idx]
	y_train = y[train_idx]
	X_cv= X[cv_idx]
	y_cv = y[cv_idx]
	
	# Create dataloaders
	train_ds = TensorDataset(X_train, y_train)
	cv_ds = TensorDataset(X_cv, y_cv)
	train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True)
	cv_dl = DataLoader(cv_ds, batch_size=batch_size, shuffle=True)
	
	return {"train": train_dl, "cv": cv_dl}
	
def train(model, optimizer, loaders, criterion, num_epochs, epoch, device=torch.device("cpu")):	
	model.train()
	losses = []
		
	# enumerate batches in epoch
	for batch_idx, (data, target) in enumerate(loaders["train"]):
		
		if batch_idx > 3: break
		
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		pred = model(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
				
def cv(model, loaders, criterion, replacement_threshold, device=torch.device("cpu")):
	model.eval()
	with torch.no_grad():
		# Only keep one loss + F1 score
		# Want equal number of train and test cases for learning curve
		keeper_idx = random.randint(0,len(loaders["cv"])-1)
		for batch_idx, (data, target) in enumerate(loaders["cv"]):  
			if batch_idx != keeper_idx: continue
			
			pred = model(data)
			loss = criterion(pred, target)
			break
	
	
	
	################# RETURN F1 SCORE
	f1 = f1_score(pred, target, replacement_threshold)
	
	print("f1",f1)
			
	return loss.item(), f1	  

from sklearn.metrics import roc_auc_score

def roc_auc(model):
	model.eval()
	with torch.no_grad():
	    y_probas = model(MemCache.corrupt_test_data)
	#auc = roc_auc_score(MemCache.target, y_probas.numpy(), average="macro")
	return y_probas

class EarlyStopping(Stopper):
    def __init__(self, metric, std=0.001, top=10, mode="min", patience=0):
        """Create the EarlyStopping object.
        Stops the entire experiment when the metric has plateaued
        for more than the given amount of iterations specified in
        the patience parameter.
        Args:
            metric (str): The metric to be monitored.
            std (float): The minimal standard deviation after which
                the tuning process has to stop.
            top (int): The number of best model to consider.
            mode (str): The mode to select the top results.
                Can either be "min" or "max".
            patience (int): Number of epochs to wait for
                a change in the top models.
        Raises:
            ValueError: If the mode parameter is not "min" nor "max".
            ValueError: If the top parameter is not an integer
                greater than 1.
            ValueError: If the standard deviation parameter is not
                a strictly positive float.
            ValueError: If the patience parameter is not
                a strictly positive integer.
        """
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be"
                             " either min or max.")
        if not isinstance(top, int) or top <= 1:
            raise ValueError("Top results to consider must be"
                             " a positive integer greater than one.")
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("Patience must be"
                             " a strictly positive integer.")
        if not isinstance(std, float) or std <= 0:
            raise ValueError("The standard deviation must be"
                             " a strictly positive float number.")
        self._mode = mode
        self._metric = metric
        self._patience = patience
        self._iterations = 0
        self._std = std
        self._top = top
        self._top_values = []

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        self._top_values.append(result[self._metric])
        if self._mode == "min":
            self._top_values = sorted(self._top_values)[:self._top]
        else:
            self._top_values = sorted(self._top_values)[-self._top:]

        # If the current iteration has to stop
        if self.has_plateaued():
            # we increment the total counter of iterations
            self._iterations += 1
        else:
            # otherwise we reset the counter
            self._iterations = 0

        # and then call the method that re-executes
        # the checks, including the iterations.
        return self.stop_all()

    def has_plateaued(self):
        return (len(self._top_values) == self._top
                and np.std(self._top_values) <= self._std)

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return self.has_plateaued() and self._iterations >= self._patience


	
def train_AE(config, reporter):
	
	print(config)
	
              
	use_cuda = config.get("use_gpu") and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	
	num_features2 = MemCache.num_features
		
	model = models.AutoEncoder(num_features2, int(config["nn_layers"]))
	model = model.to(device)
	
	if config["optimizer"] == "adam":
		optimizer = optim.Adam(
			model.parameters(),
			lr=config["lr"],
			weight_decay=config["weight_decay"]
			)
	elif config["optimizer"] == "sgd":
		optimizer = optim.SGD(
			model.parameters(),
			lr=config["lr"],
			weight_decay=config["weight_decay"]
			)	
			
	criterion = nn.BCELoss(reduction='sum')
	loaders = cv_dataloader(config["batch_size"], num_features2, config["kfolds"])
		
	for epoch in range(config["num_epochs"]):
	
		model.train()
		losses = []
			
		# enumerate batches in epoch
		for batch_idx, (data, target) in enumerate(loaders["train"]):
			
			# DELETE EVENTUALLY
			if batch_idx > 9: break
			
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			pred = model(data)
			loss = criterion(pred, target)
			loss.backward()
			optimizer.step()
			
			# SET T0 100 EVENTUALLY
			if (batch_idx+1) % 10 == 1:
				 
				train_loss = loss.item()
				train_f1 = f1_score(pred, target, config["replacement_threshold"])
				test_loss, test_f1 = cv(model, loaders, criterion, config["replacement_threshold"], device)
				y_probas = roc_auc(model)
				# SAVE Y_PROBA
				#print("auc",auc)
				reporter(test_f1=test_f1, train_f1=train_f1, test_loss=test_loss, train_loss=train_loss) #, auc_score=auc)	
		
		torch.save(model.state_dict(), "./model.pt")
		torch.save(y_probas, "./y_probas.pt")	
		
		
		
		
		
		
		
	