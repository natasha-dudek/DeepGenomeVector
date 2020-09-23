from copy import deepcopy
import gc
import os
import psutil
import random
import time
import sys

import argparse
from argparse import Namespace
import numpy as np
np.seterr(all="raise")
from filelock import FileLock
import pandas as pd
import pickle
import ray
from ray import tune
from ray.tune import track
from ray.tune.stopper import Stopper
import sklearn as sk
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Binarizer
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from genome_embeddings import models
from genome_embeddings import data_viz
from genome_embeddings import evaluate
from genome_embeddings import models
from genome_embeddings import train_test
from genome_embeddings import util
from genome_embeddings import models 

#os.system("rm file_tout")
#os.system("rm file_terr")
#
#sys.stdout = open('file_tout', 'w')
#sys.stderr = open('file_terr', 'w')

class MemCache:
	###########################
	# TO RUN ON CC:
	DATA_FP = "/home/ndudek/projects/def-dprecup/ndudek/"
	train_data=torch.load(DATA_FP+"corrupted_train_2020-09-09.pt")
	test_data=torch.load(DATA_FP+"corrupted_test_2020-09-09.pt")
	#df_train_data = pd.DataFrame(train_data)

	#genome_to_tax = np.load(DATA_FP+'genome_to_tax.npy', allow_pickle='TRUE').item()
	#genome_idx_train = torch.load(DATA_FP+"genome_idx_train_07-17-20.pt")
	#genome_idx_test = torch.load(DATA_FP+"genome_idx_test_07-17-20.pt")
  
#	df, cluster_names = util.load_data(DATA_FP, "kegg")
#	genome_to_num ={}
#	for i,genome in enumerate(df.index):
#		genome_to_num[genome] = i
#	num_to_genome = {v: k for k, v in genome_to_num.items()}
	
	# To make predictions on (ROC + AUC)
	num_features = int(train_data.shape[1]/2)
	#tensor_test_data = torch.tensor(test_data).float()
	corrupt_test_data = test_data[:,:num_features]
	#target = test_data[:,num_features:].numpy() # no grad
	
#	train_data = torch.Tensor(train_data)
#	test_data = torch.Tensor(test_data)

#	# split X and y
	X = train_data[:,:num_features]  #corrupted genomes in first half of matrix columns
	y = train_data[:,num_features:]  #uncorrupted in second half of matrix columns
	
	###########################
	# TO RUN ON LAPTOP
	
#	DATA_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_embeddings/data/'
#	train_data = torch.load("/Users/natasha/Desktop/corrupted_train_mini.pt")
#	test_data = torch.load("/Users/natasha/Desktop/corrupted_test_mini.pt")
#	#df, cluster_names = util.load_data(DATA_FP, "kegg")
#	# To make predictions on (ROC + AUC)
#	num_features = int(train_data.shape[1]/2)
#	tensor_test_data = torch.tensor([i.numpy() for i in test_data]).float()
#	corrupt_test_data = tensor_test_data[:,:num_features]
#	target = tensor_test_data[:,num_features:].detach().numpy()
#	
#	X = train_data[:,:num_features]  #corrupted genomes in first half of matrix columns
#	y = train_data[:,num_features:]  #uncorrupted in second half of matrix columns
		
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
			pred_arr = i.detach().numpy()	 
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
	if average: avg_f1 -- average F1 score
	"""
	binarized_preds = binarize(pred_non_bin, replacement_threshold)
	
	f1s = []
	for i in range(0,len(binarized_preds)):
		f1 = sk.metrics.f1_score(target.data.numpy()[i], binarized_preds[i])
		f1s.append(f1)
	
	return sum(f1s)/len(f1s)

def cv_dataloader(batch_size, num_features, k):

	"""
	Creates training dataloader w/ k-folds for cross validation
	Note: only creates 1 split regardless of # k -- improves training speed (method of skorch CVSplit)
	
	Arguments:
	batch_size (int) -- batch size for training set
	num_features (int) -- number of features / genes
	k (int) -- number of folds
	Note: accesses training and test data via MemCache.training_data and MemCache.test_data
	
	Returns:
	dict of DataLoaders -- train and cross-validation dataloaders
		dict["train"] -- training dataloader, batch_size = batch_size
		dict["cv"] -- cross-validation dataloader, batch_size = 1000 (hard-coded)
	"""
	# load train data from memory (saves time and, more importantly, space)
	X = MemCache.X # corrupted genomes
	y = MemCache.y # uncorrupted genomes
		
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
	batch_size_cv = 1000 # always test on CV set of size 1000
	train_ds = TensorDataset(X_train, y_train)
	cv_ds = TensorDataset(X_cv, y_cv)
	train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True)
	cv_dl = DataLoader(cv_ds, batch_size=batch_size_cv, shuffle=True)
	
	return {"train": train_dl, "cv": cv_dl}
				
def cv(model, loaders, replacement_threshold, device=torch.device("cpu")):
	"""
	Evaluate model on cross-validation set
	
	Arguments:
	model -- pytorch model
	loaders (dict of DataLoaders) -- dictionary of dataloader, here we use loaders["cv"]
	criterion -- pytorch criterion for evaluating model (e.g.: loss)
	replacement_threshold -- threshold for converting predicted probabilities to 1's or 0's
	device (str) -- cpu or cuda
	
	Returns:
	loss (float) -- Loss on a randomly selected batch from the test set
	f1 (float) -- F1 score on a randomly selected batch from the test set (same one as for loss)
	"""
	
	model.eval()
	with torch.no_grad():
		# Only calculate loss + F1 score on one batch of CV set (increase training speed)
		keeper_idx = random.randint(0,len(loaders["cv"])-1)
		for batch_idx, (corrupt_data, target) in enumerate(loaders["cv"]):  
			if batch_idx != keeper_idx: continue
			pred, mu, logvar = model.forward(corrupt_data)
			loss = vae_loss(pred, target, mu, logvar)
			break

	f1 = f1_score(pred, target, replacement_threshold)
				
	return loss, f1	  

def roc_auc(model):
	"""
	Create probability predictions
	
	Arguments:
	model -- pytorch model
	Note: uses corrupted test_data from MemCache
	
	Returns:
	y_probas -- predicted probabilities
	"""
	model.eval()
	with torch.no_grad():
		y_probas = model(MemCache.corrupt_test_data)
	#auc = roc_auc_score(MemCache.target, y_probas.numpy(), average="macro")
	return y_probas

class EarlyStopping(Stopper):
	"""
	Implements fancy early stopping for hyperparameter tuning
	
	Copy-pasted from a newer version of Ray Tune that isn't compatible with scientific computing package versions on Compute Canada
	"""
	
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

def auto_garbage_collect(pct=80.0):
	"""
	If memory usage is high, call garbage collector
	"""
	if psutil.virtual_memory().percent >= pct:
		gc.collect()

def vae_loss(pred, target, mu, logvar):
	"""
	Compute VAE loss for a given batch of genomes to be binary cross entropy (BCE) + Kullback-Leibler divergence (KLD)

	Note: 
	BCE tries to make the reconstruction as accurate as possible
	KLD tries to make the learned distribtion as similar as possible to the unit Gaussian 
	
	Arguments:
	pred (tensor) -- prediction for which bits should be on in each genome vector
	target (tensor) -- ground truth for which bits should be on in each genome vector
	mu (tensor) -- mean for each latent dimension of the code layer
	logvar (tensor) -- variance for each latent dimension of the code layer
	
	Returns:
	BCE + KLD (tensor) -- loss for one batch of genomes
	"""

	# Reading: https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/#what-is-a-variational-autoencoder
	# Binary cross entropy: how well do output + target agree
	BCE = F.binary_cross_entropy(pred, target, reduction='sum')
	#print("logvar.max()", logvar.max(),"logvar.exp()",logvar.exp().max())
	# how much does the learned distribution vary from the unit Gaussian
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 10000
	# Normalize KLD to prevent KLD from dominating loss
	# Normalize by same number of elements as there are in the reconstruction
	#KLD /= pred.shape[0]*pred.shape[1]
	
	loss = KLD + BCE
	loss = torch.min(loss, 1000000000*torch.ones_like(loss))
#	KLD = torch.min(KLD, 1000000*torch.ones_like(KLD))
#	BCE = torch.min(BCE, 1000000*torch.ones_like(BCE))
	
	#print("loss", loss.max())
	
	return loss
	
def train_AE(config, reporter):
	"""
	Train autoencoder, save model and y_probas 
	
	Arguments:
	config (dict) -- contains parameter and hyperparameter settings for a given trial
		nn_layers -- number of layers in neural net
		weight_decay -- weight_decay
		batch_size - batch_size to use for training data loader
		kfolds -- number of folds for K-fold cross validation
		num_epochs -- number of epochs for which to train
	reporter (progress_reporter) -- ray tune progress reporter
	
	Returns:
	None
	"""	
	use_cuda = config.get("use_gpu") and torch.cuda.is_available()
	device = torch.device("cpu") #"cuda" if use_cuda else "cpu")
	
	num_features2 = MemCache.num_features
	
	model = models.VariationalAutoEncoder(num_features2, int(config["nn_layers"]))
	model = model.to(device)
	model.train()
	
	optimizer = optim.AdamW(
			model.parameters(),
			lr=config["lr"],
			weight_decay=config["weight_decay"]
			)
		
	loaders = cv_dataloader(int(config["batch_size"]), num_features2, config["kfolds"])
			
	for epoch in range(config["num_epochs"]):
		losses = []
			
		# enumerate batches in epoch
		for batch_idx, (corrupt_data, target) in enumerate(loaders["train"]):
			corrupt_data, target = corrupt_data.to(device), target.to(device)
			optimizer.zero_grad()
			pred, mu, logvar = model.forward(corrupt_data)
			loss = vae_loss(pred, target, mu, logvar)
			
#			if torch.isnan(loss.max()):
#				print("Loss IS A NAN")
					 
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

			optimizer.step()
			
			# Every 100 batches, take stock of how well things are going
			if (batch_idx+1) % 100 == 1:
				train_loss = loss.item()
				train_f1 = f1_score(pred, target, config["replacement_threshold"])
				test_loss, test_f1 = cv(model, loaders, config["replacement_threshold"], device)
				y_probas = roc_auc(model)
				reporter(test_f1=test_f1, train_f1=train_f1, test_loss=test_loss, train_loss=train_loss) #, auc_score=auc)	
				model.train()
				
			sys.stdout.flush()
			sys.stderr.flush()
							
	# save results (will save to tune results dir)			  
	torch.save(model.state_dict(), "./model.pt")
	torch.save(y_probas, "./y_probas.pt")	
	# if memory usage is high, may be able to free up space by calling garbage collect
	auto_garbage_collect()		 
		
		
		
		
		
		
	