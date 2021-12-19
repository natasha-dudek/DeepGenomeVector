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
#	###########################
#	# TO RUN ON CC:
#	DATA_FP = "/home/ndudek/projects/def-dprecup/ndudek/hp_tuning_04-12-2020/"
#	train_data=torch.load(DATA_FP+"corrupted_train_2020-12-04_10mods.pt")
#	test_data=torch.load(DATA_FP+"corrupted_test_2020-12-04_10mods.pt")
#	# To make predictions on (ROC + AUC)
#	num_features = int(train_data.shape[1]/2)
#
#	corrupt_test_data = test_data[:,:num_features]
#
#	# split X and y
#	X = train_data[:,:num_features]  #corrupted genomes in first half of matrix columns
#	y = train_data[:,num_features:]  #uncorrupted in second half of matrix columns
#	
#	##########################
#	 TO RUN ON LAPTOP
	# done 5, 0
	train_data = torch.load("/Users/natasha/Desktop/vae/corrupted_train_2021-01-05_40mods.pt")
	test_data = torch.load("/Users/natasha/Desktop/vae/corrupted_test_2021-01-05_40mods.pt")
	
	print("loaded train + test data")
	
	# To make predictions on (ROC + AUC)
	num_features = int(train_data.shape[1]/2)
	corrupt_test_data = test_data[:,:num_features]
	
	X = train_data[:,:num_features]  #corrupted genomes in first half of matrix columns
	y = train_data[:,num_features:]  #uncorrupted in second half of matrix columns
		
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
	sum(f1s)/len(f1s) -- average F1 score for batch
	"""
	binarized_preds = binarize(pred_non_bin, replacement_threshold)
	
	f1s = []
	recalls = []
	precisions = []
	
	for i in range(0,len(binarized_preds)):
		f1 = sk.metrics.f1_score(target.data.numpy()[i], binarized_preds[i], zero_division=0)
		f1s.append(f1)
		
#		recall = sk.metrics.recall_score(target.data.numpy()[i], binarized_preds[i])
#		recalls.append(recall)
#		
#		precision = sk.metrics.precision_score(target.data.numpy()[i], binarized_preds[i])
#		precisions.append(recall)
		
		#zero_rows = binarized_preds[i].sum(dim = 1) == 0).squeeze()
#		n_zeros = np.count_nonzero(binarized_preds[i]==0)
#		n_genes = len(binarized_preds[i])
#		n_ones = n_genes - n_zeros
#		print(n_zeros, n_ones, n_genes)
		#a,b,c,d = sk.metrics.confusion_matrix(target.data.numpy()[i], binarized_preds[i]).flatten()
#		mili = [a,b,c,d]
#		if 0 in mili:
#			print(mili)
	
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

def cv_dataloader_SINGLE(batch_size, num_features, k, train_data, test_data):

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
	
	# To make predictions on (ROC + AUC)
	num_features = int(train_data.shape[1]/2)
	corrupt_test_data = test_data[:,:num_features]
	
	X = train_data[:,:num_features]  #corrupted genomes in first half of matrix columns
	y = train_data[:,num_features:]  #uncorrupted in second half of matrix columns
	   
		
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
	
				
def cv_vae(model, loaders, replacement_threshold, device=torch.device("cpu")):
	"""
	Evaluate model on cross-validation set
	
	Arguments:
	model -- pytorch model
	loaders (dict of DataLoaders) -- dictionary of dataloader, here we use loaders["cv"]
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
			loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
			break
	f1 = f1_score(pred, target, replacement_threshold)
				
	return loss, f1	  

def cv_dae(model, loaders, replacement_threshold, criterion, device=torch.device("cpu")):
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
			pred = model.forward(corrupt_data)
			loss = criterion(pred, target)
			break

	f1 = f1_score(pred, target, replacement_threshold)
				
	return loss, f1	  

#
#def roc_auc(model):
#	"""
#	Create probability predictions
#	
#	Arguments:
#	model -- pytorch model
#	Note: uses corrupted test_data from MemCache
#	
#	Returns:
#	y_probas -- predicted probabilities
#	"""
#	model.eval()
#	with torch.no_grad():
#		y_probas = model(MemCache.corrupt_test_data)
#	#auc = roc_auc_score(MemCache.target, y_probas.numpy(), average="macro")
#	return y_probas

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
	# Multiply KLD by factor of 10,000 such that it is on the same order of magnitude as BCE
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
	
	loss = KLD + BCE
	loss = torch.min(loss, 1000000000*torch.ones_like(loss))

	return loss, KLD, BCE
	
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
	print(num_features2)
	print('config["replacement_threshold"]',config["replacement_threshold"])
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
			loss, KLD, BCE = vae_loss(pred, target, mu, logvar)					 
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
			optimizer.step()
			
			# Every 100 batches, take stock of how well things are going
			if (batch_idx+1) % 100 == 1:
				train_loss = loss.item()
				train_f1 = f1_score(pred, target, config["replacement_threshold"])
				# note that "test_f1 / loss" is actually for a CV fold
				test_loss, test_f1 = cv_vae(model, loaders, config["replacement_threshold"])
				#y_probas = roc_auc(model)
				reporter(test_f1=test_f1, train_f1=train_f1, test_loss=test_loss, train_loss=train_loss) #, auc_score=auc)	
				model.train()
				
			sys.stdout.flush()
			sys.stderr.flush()
							
	# save results (will save to tune results dir)			  
	torch.save(model.state_dict(), "./model.pt")
	#torch.save(y_probas, "./y_probas.pt")	
	# if memory usage is high, may be able to free up space by calling garbage collect
	auto_garbage_collect()		 
		
def train_single_dae(nn_layers, weight_decay, lr, batch_size, kfolds, num_epochs, replacement_threshold):
	"""
	Train single run of autoencoder, save model and y_probas 
	
	Arguments:
	nn_layers -- number of layers in neural net
	weight_decay -- weight_decay
	lr -- learning rate
	batch_size - batch_size to use for training data loader
	kfolds -- number of folds for K-fold cross validation
	num_epochs -- number of epochs for which to train
	replacement_threshold -- probability thresh after which to convert bit to 1 vs 0
	
	Returns:
	None
	"""	
	device = torch.device("cpu") #"cuda" if use_cuda else "cpu")
	
	num_features2 = MemCache.num_features
	
	model = models.AutoEncoder(num_features2, nn_layers)
	model = model.to(device)
	model.train()
	
	criterion = nn.BCELoss(reduction='sum')
	
	optimizer = optim.AdamW(
			model.parameters(),
			lr=lr,
			weight_decay=weight_decay
			)
		
	loaders = cv_dataloader(batch_size, num_features2, kfolds)
			
	for epoch in range(num_epochs):
		losses = []
			
		# enumerate batches in epoch
		for batch_idx, (corrupt_data, target) in enumerate(loaders["train"]):
			corrupt_data, target = corrupt_data.to(device), target.to(device)
			optimizer.zero_grad()
			pred, mu, logvar = model.forward(corrupt_data)
			loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
			optimizer.step()
			
			# Every 100 batches, take stock of how well things are going
			if (batch_idx+1) % 100 == 1:
				train_loss = loss.item()
				train_f1 = f1_score(pred, target, config["replacement_threshold"])
				test_loss, test_f1 = cv(model, loaders, config["replacement_threshold"], device)
				#y_probas = roc_auc(model)
				reporter(test_f1=test_f1, train_f1=train_f1, test_loss=test_loss, train_loss=train_loss)
				model.train()
				
			sys.stdout.flush()
			sys.stderr.flush()
							
	# save results (will save to tune results dir)			  
	#torch.save(model.state_dict(), "./model.pt")
	#torch.save(y_probas, "./y_probas.pt")	
	# if memory usage is high, may be able to free up space by calling garbage collect
	auto_garbage_collect()				 


def min_hamming_dist(pred, train_data):
	
	n_rows = train_data.shape[0]
	n_col = train_data.shape[1]
	pred_tensor = np.tile(pred, n_rows).reshape(n_rows,n_col)	
	sklearn.metrics.hamming_loss(train_data, pred_tensor)

	
		
		
def train_single_vae(nn_layers, weight_decay, lr, batch_size, kfolds, num_epochs, replacement_threshold, train_data, test_data):
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
	print("11:32")
	kld0 = []
	kld1 = []
	bce0 = []
	bce1 = []
	train_losses = []
	test_losses = []
	train_f1s = []
	test_f1s = []
	
	device = torch.device("cpu") #"cuda" if use_cuda else "cpu")
	
	num_features = int(train_data.shape[1]/2)
	
	
	#print("num_features2",num_features2)
	
	model = models.VariationalAutoEncoder(num_features, nn_layers)
	model = model.to(device)
	model.train()
		
	optimizer = optim.AdamW(
			model.parameters(),
			lr=lr,
			weight_decay=weight_decay
			)
	
		
	loaders = cv_dataloader_SINGLE(batch_size, num_features, kfolds, train_data, test_data)
	#loaders = cv_dataloader(batch_size, num_features, kfolds)
			
	for epoch in range(num_epochs):
		
		  
		# enumerate batches in epoch
		for batch_idx, (corrupt_data, target) in enumerate(loaders["train"]):
			corrupt_data, target = corrupt_data.to(device), target.to(device)
			optimizer.zero_grad()
			pred, mu, logvar = model.forward(corrupt_data)
			
			loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
			
			kld0.append(KLD)
			bce0.append(BCE)
		  
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

			
			
			# Every 100 batches, take stock of how well things are going
			if batch_idx % 100 == 0:
				train_f1 = f1_score(pred, target, replacement_threshold)
				test_loss, test_f1 = cv_vae(model, loaders, replacement_threshold)
				train_losses.append(loss.item())
				test_losses.append(test_loss.item())
				train_f1s.append(train_f1)
				test_f1s.append(test_f1)
				print("epoch",epoch,"batch",batch_idx)
				print("train_loss",loss.item(), "train_f1",train_f1, "test_loss",test_loss.item(), "test_f1",test_f1) #, auc_score=auc)	
				model.train()
				
			
			optimizer.step()
				
			sys.stdout.flush()
			sys.stderr.flush()
	
	#y_probas = roc_auc(model)
							
	# save results (will save to tune results dir)			  
	#torch.save(model.state_dict(), "/Users/natasha/Desktop/model.pt")
	#torch.save(y_probas, "/Users/natasha/Desktop/y_probas.pt")	
	# if memory usage is high, may be able to free up space by calling garbage collect
	auto_garbage_collect()  
	
	return kld0, bce0, train_losses, test_losses, train_f1s, test_f1s, model

def hpo_vae(nn_layers, weight_decay, lr, batch_size, kfolds, num_epochs, replacement_threshold, train_data, test_data):
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

	train_losses = []
	test_losses = []
	train_f1s = []
	test_f1s = []
	
	device = torch.device("cpu") #"cuda" if use_cuda else "cpu")
	
	num_features = MemCache.num_features
	
	
	#print("num_features2",num_features2)
	
	model = models.VariationalAutoEncoder(num_features, int(config["nn_layers"]))
	model = model.to(device)
	model.train()
		
	optimizer = optim.AdamW(
			model.parameters(),
			lr=config["lr"],
			weight_decay=config["weight_decay"]
			)
	
		
	#loaders = cv_dataloader_SINGLE(batch_size, num_features, kfolds, train_data, test_data)
	loaders = cv_dataloader(int(config["batch_size"]), num_features, config["kfolds"])
			
	for epoch in range(config["num_epochs"]):
		losses = []
		  
		# enumerate batches in epoch
		for batch_idx, (corrupt_data, target) in enumerate(loaders["train"]):
			corrupt_data, target = corrupt_data.to(device), target.to(device)
			optimizer.zero_grad()
			pred, mu, logvar = model.forward(corrupt_data)
			
			loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
		  
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

			
			
			# Every 100 batches, take stock of how well things are going
			if batch_idx % 100 == 0:
				train_loss = loss.item()
				train_f1 = f1_score(pred, target, config["replacement_threshold"])
				test_loss, test_f1 = cv_vae(model, loaders, replacement_threshold)
				#y_probas = roc_auc(model)
				reporter(test_f1=test_f1, train_f1=train_f1, test_loss=test_loss, train_loss=train_loss) #, auc_score=auc)	
				model.train()
				
			
			optimizer.step()
				
			sys.stdout.flush()
			sys.stderr.flush()
			
	# if memory usage is high, may be able to free up space by calling garbage collect
	auto_garbage_collect()  
