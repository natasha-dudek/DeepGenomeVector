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

DATA_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_embeddings/data/'

class MemCache:
	train_data = torch.load(DATA_FP+"corrupted_train.pt")
	test_data = torch.load(DATA_FP+"corrupted_test.pt")
	genome_idx_train = torch.load(DATA_FP+"genome_idx_train.pt")
	genome_idx_test = torch.load(DATA_FP+"genome_idx_test.pt")
	
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
	t0 = time.time()
#	train_data = torch.load(DATA_FP+"corrupted_train.pt")
#	test_data = torch.load(DATA_FP+"corrupted_test.pt")
#	genome_idx_train = torch.load(DATA_FP+"genome_idx_train.pt")
#	genome_idx_test = torch.load(DATA_FP+"genome_idx_test.pt")
	train_data = MemCache.train_data
	test_data = MemCache.test_data
	genome_idx_train = MemCache.genome_idx_train
	genome_idx_test = MemCache.genome_idx_test

	print("time to load data from file", time.time() - t0)
	t1 = time.time()
	loaders = util.dataloaders(train_data, test_data, batch_size, batch_size, num_features)
	print("time to create dataloaders", time.time() - t1)
	
	return loaders
	
def train_updates(epoch, num_epochs, idx, batch_per_epoch, loss):
	print('\r Training epoch {}/{}, batch {}/{}, {} loader, \tLoss: {:.2f}'.format(
						epoch + 1, # epoch just finished, 1-indexed
						num_epochs, # total number of epochs
						idx, # batch just finished
						batch_per_epoch, # total number batches per epoch
						loss 
						)
						)	 
	
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
	
	print ("done training")
#		if batch_idx % 100 == 1:
#			# print update
#			train_updates(epoch, num_epochs, batch_idx, len(loaders['train']), loss.cpu().data.item())
				
	#return loss.item()
			
def test(model, loaders, criterion, replacement_threshold, device=torch.device("cpu")):
	model.eval()
	with torch.no_grad():
		# Only keep one loss + F1 score
		# Want equal number of train and test cases for learning curve
		keeper_idx = random.randint(0,len(loaders['test'])-1)
		for batch_idx, (data, target) in enumerate(loaders["test"]):  
			if batch_idx != keeper_idx: continue
			
			pred = model(data)
			loss = criterion(pred, target)
			break
	
	
	
	################# RETURN F1 SCORE
	f1 = f1_score(pred, target, replacement_threshold)
	
	print("f1",f1)
			
	return loss.item(), f1	  

from genome_embeddings import models 

def train_AE(config, reporter):

	#print("launching train_AE, lr:",config["lr"],"wd:",config["weight_decay"])
	
	use_cuda = config.get("use_gpu") and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	
	num_epochs = int(config["num_epochs"])
	batch_size = int(config["batch_size"])
	print("batch_size",batch_size)
	num_features = int(MemCache.train_data.shape[1]/2)
	model = models.AutoEncoder(num_features)
	model = model.to(device)
	loaders = get_dataloader(DATA_FP, batch_size, num_features)

	optimizer = optim.Adam(
			model.parameters(),
			lr=config.get("lr", 0.001),
			weight_decay=config.get("weight_decay", 0.1))
			
	criterion = nn.BCELoss(reduction='sum')
	
	losses = {"train":[], "test":[]}
	f1s = {"train":[], "test":[]}
	
	# enumerate epochs
	for epoch in range(num_epochs):
		#print("Beginning training for epoch",epoch)
		#train_losses, train_f1 = train(model, optimizer, loaders, criterion, num_epochs, epoch, device)
		train(model, optimizer, loaders, criterion, num_epochs, epoch, device)
		test_losses, test_f1 = test(model, loaders, criterion, config["replacement_threshold"], device)
		
	reporter(mean_accuracy=test_f1)	
		
#		losses["train"].append(train_losses)
#		losses["test"].append(test_losses)
#		losses["train"].append(train_f1s)
#		losses["test"].append(test_f1s)
		
#	return losses, f1s
