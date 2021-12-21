from collections import defaultdict
from copy import deepcopy
import pickle
import random
import re

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import numpy as np
import pandas as pd
import pylab as P
from pingouin import anova
from scipy import stats
import seaborn as sns
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc, confusion_matrix, hamming_loss, roc_curve
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Binarizer
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import torch

from genome_embeddings import pre_process

def eval_binarize(pred, replacement_threshold):
	"""
	Convert tensor of probabilites to binary values, using replacement_threshold
	
	Arguments:
		pred (tensor) -- predictions with probability values
		replacement_threshold (float) -- threshold at which to replace pred scores with 0 or 1
	
	Returns:
		(tensor) -- predictions with binary values
	"""
	return torch.LongTensor(Binarizer(threshold=replacement_threshold).fit_transform(pred))

def confusion(uncorrupted, binary_pred):
	"""
	Calculate TNs, FPs, FNs, TPs
	
	Arguments: 
		corrupted (tensor) -- corrupted data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
		
	Returns:
		tns (list) -- number of true negatives for each genome
		fps (list) -- number of false positives for each genome
		fns (list) -- number of false negatives for each genome
		tps (list) -- number of true positives for each genome
	"""
	tns = []
	fps = []
	fns = []
	tps = []
	
	for idx, row in enumerate(uncorrupted):
		tn, fp, fn, tp = confusion_matrix(row, binary_pred[idx]).ravel()
		tns.append(tn)
		fps.append(fp)
		fns.append(fn)
		tps.append(tp)
		
	total = sum(tns+fps+fns+tps)
	
	# print percentages of tns, fps, fns, tps
	p_tns = round(sum(tns)/total*100,2)
	p_fps = round(sum(fps)/total*100,2)
	p_fns = round(sum(fns)/total*100,2)
	p_tps = round(sum(tps)/total*100,2)
	print('The percentage of TNs, FPs, FNs, and TPs, respectively, is:',p_tns, p_fps, p_fns, p_tps)	
	return tns, fps, fns, tps

def kld_vs_bce(kld, bce):
	"""
	Generate scatterplot showing KLD and BCE loss vs experience
	
	Arguments:
		kld (list) -- kld values over training
		bce (list) -- bce values over training
	
	Returns:
		matplotlib.Figure
	"""
	x = [i for i in range(len(kld))]
	kld = [float(i) for i in kld]
	bce = [float(i) for i in bce]
	fig = plt.plot()
	plt.scatter(x,kld, c='b', marker='.', label='KLD')
	plt.scatter(x,bce, c='r', marker='.', label='BCE')
	plt.legend(loc='upper right')
	plt.xlabel("Experience")
	plt.ylabel("Loss")
	plt.yscale('log')

	return fig

def pixel_diagram(corrupted, uncorrupted, idx, model, f1s, tns, fps, fns, tps, binary_pred):
	"""
	Plot a pixel diagram (heatmap) visualizing the number of TNs, FPs, FNs, TPs
	
	Arguments:
		corrupted (tensor) -- corrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		uncorrupted  (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		idx (int) -- index of corrupted genome to investigate
		model (genome_embeddings.models.VariationalAutoEncoder) -- trained VAE model
		f1s (list) -- test F1 scores
		tns (list) -- number of true negatives for each genome
		fps (list) -- number of false positives for each genome
		fns (list) -- number of false negatives for each genome
		tps (list) -- number of true positives for each genome
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
	
	Returns:
		matplotlib.Figure
	"""
	colours = ['black', 'green', 'magenta', 'yellow', 'white']
	cmap_name = 'my_list'

	# Get corrupted and predicted genome vectors
	corr_genome = corrupted[idx]
	true_genome = uncorrupted[idx]
	binary_pred = binary_pred[idx]
	
	# set up dimensions of pixel rectangle
	n_features = int(corrupted.shape[1])
	n_extension = 100*99 - n_features
	n_rows = 99
	n_cols = 100
	
	# Format corrupted version of genome
	corrupted = corrupted[idx].tolist()
	corrupted.extend([4] * n_extension) # 100*100 - n_features
	corrupted = np.reshape(corrupted, (n_rows, n_cols))
	cm = LinearSegmentedColormap.from_list(cmap_name, colours, N=len(colours))
	print("Corrupted -- Genes on:",str(int(sum(sum(corrupted)))),"Genes off:",str(int(n_features - sum(sum(corrupted))))) 
	
	# Format uncorrupted version of genome
	uncorrupted = uncorrupted[idx].tolist()
	uncorrupted.extend([4] * n_extension) # 100*100 - n_features
	uncorrupted = np.reshape(uncorrupted, (n_rows, n_cols)) 
	print("Uncorrupted -- Genes on:",str(int(sum(sum(uncorrupted)))),"Genes off:",str(int(n_features - sum(sum(uncorrupted))))) 

	tn = tns[idx]
	fp = fps[idx]
	fn = fns[idx]
	tp = tps[idx]
	print("Generated -- TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp)
	print("The F1 score for this reconstruction was",f1s[idx])
	
	# Colour pixels
	colour_pred = []
	for i in zip(binary_pred, corr_genome, true_genome):
		if i[0] == i[2] == 1: # TP
			colour_pred.append(1) 
		elif i[0] == i[2] == 0: # TN
			colour_pred.append(0) 
		elif i[0] == 0 and i[2] == 1: # FN
			colour_pred.append(2)
		else: # FP
			colour_pred.append(3)
	
	# Plot		
	colour_pred.extend([4] * n_extension) # 100*100 - n_features
	colour_pred = np.reshape(colour_pred, (n_rows, n_cols))  
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
	ax1.imshow(uncorrupted, cmap=cm, interpolation='nearest')
	ax2.imshow(corrupted, cmap=cm, interpolation='nearest')  
	ax3.imshow(colour_pred, cmap=cm, interpolation='nearest')
	ax1.set_title("Original (uncorrupted)")
	ax2.set_title("Corrupted")
	ax3.set_title("Generated")
	
	# turn off tick labels and markers
	for i in (ax1, ax2, ax3):
		i.set_xticks([])
		i.set_yticks([])
	
	return fig
	
def learning_curve(train_losses, test_losses, train_f1s, test_f1s):
	"""
	Plots optimization (loss) and performance (F1) learning curves 
		
	Arguments:
		train_losses (list) -- training losses (KLD + BCE)
		test_losses (list) -- test losses (KLD + BCE)
		train_f1s (list) -- training F1	scores
		test_f1s (list) -- test F1 scores
	
	Returns:
		matplotlib.Figure
	"""
	
	plt.rcParams.update({'font.size': 16})
	
	x_losses = [*range(len(train_losses))]
	
	fig, axs = plt.subplots(1, 2, figsize=(10, 5))
	
	axs[0].set_title("Optimization Learning Curve")
	axs[1].set_title("Performance Learning Curve")
	
	axs[0].set_ylim(10**4,10**7)
	axs[1].set_ylim(0,1)
	
	axs[0].plot(x_losses, train_losses, marker='.', c='#3385ff', label='Training', markersize=5)
	axs[0].plot(x_losses, test_losses, marker='.', c='#ff6666', label='CV', markersize=5)
	
	axs[1].plot(x_losses, train_f1s, marker='.', c='#3385ff', label='Training', markersize=5)
	axs[1].plot(x_losses, test_f1s, marker='.', c='#ff6666', label='CV', markersize=5)
	
	axs[0].set_xlim(-5,x_losses[-1]+5)
	axs[1].set_xlim(-5,x_losses[-1]+5)
	
	axs[0].set_ylabel('Loss (KLD + BCE)')
	axs[0].semilogy()
	axs[1].set_ylabel('F1 score')
	
	axs[0].set_xlabel('Experience')
	axs[1].set_xlabel('Experience')
	
	axs[1].axhline(y=max(test_f1s), color='r', dashes=(1,1))
	print("max F1 score", max(test_f1s))
	
	axs[0].legend(loc="upper right")
	
	plt.tight_layout()
	
	return fig

def baseline1(corrupted_train, tla_to_mod_to_kos, tnum_to_kos, tla_to_tnum, c_train_genomes, corrupted_test):
	"""
	Create baseline1 predictions: generate genome vectors by completely randomly turn on n bits, where n = a randomly selected number of genes encoded by a real genome from the training set
	
	Arguments:
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
		tnum_to_kos (dict) -- maps tnums to KOs encoded by that genome, e.g.: 'T00001': [K00001, ... 'K0000N']
		tla_to_tnum (dict) -- for each genome, converts tla to tnum
		c_train_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
		corrupted_test (tensor) -- corrupted test data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not

	Returns:
		baseline1 (tensor) -- baseline1 predictions. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
	"""
	# Use training set to get stats about gene count dist.
	def gene_dist(tla_to_mod_to_kos, tla_to_kos, tla_to_tnum, c_train_genomes):
		gene_counts = []
		for org in tla_to_mod_to_kos:
			tnum = tla_to_tnum[org]
			if org in c_train_genomes and len(tla_to_mod_to_kos[org]) >= 10 and len(tla_to_kos[tnum]) >= 400: # make sure org is in training set, not test
				if len(tla_to_kos[tnum]) < 10:
					print()
					print (tnum, len(tla_to_kos[tnum]))
					print(len(tla_to_mod_to_kos[org]))
				gene_counts.append(len(tla_to_kos[tnum]))
		return gene_counts
	gene_counts = gene_dist(tla_to_mod_to_kos, tla_to_kos, tla_to_tnum, c_train_genomes)

	n_features = int(corrupted_test.shape[1]/2)
	baseline1 = torch.zeros_like(corrupted_test)
		
	for i in range(baseline1.shape[0]):
		# get idx of on bits in corrupted vector 
		on_pos = [int(s) for s in (corrupted_test[i,:] == 1).nonzero()]
		# create vector of all idxs in tensor
		all_pos = [s for s in range(n_features)]
		# get idxs not already on by taking difference of above two vectors
		leftover = [s for s in all_pos if s not in on_pos ]
		# determine how many genes we want to be on
		n_on = random.choice(gene_counts)  
		# randomly select n_on - len(on_pos) more genes
		g = random.sample(leftover, n_on - len(on_pos))

		new = g + on_pos
		baseline1[i,:][new] = 1

	return baseline1.long()

def baseline2(corrupted_train, tla_to_mod_to_kos, tnum_to_kos, tla_to_tnum, c_train_genomes, corrupted_test):
	"""
	Create baseline2 predictions: Generate genome vectors by randomly turn on n bits with the highest probability of being on across the entire training set. n = a randomly selected number of genes encoded by a real genome from the training set
	
	Arguments:
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
		tnum_to_kos (dict) -- maps tnums to KOs encoded by that genome, e.g.: 'T00001': [K00001, ... 'K0000N']
		tla_to_tnum (dict) -- for each genome, converts tla to tnum
		c_train_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
		corrupted_test (tensor) -- corrupted test data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		
	Returns:
		baseline2 (tensor) -- baseline2 predictions. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
	"""

	n_features = int(uncorrupted_train.shape[1]/2)
	# Use training set to calculate stats about prob bits being on
	uncorrupted = corrupted_train[:,n_features:] # uncorrupted
	per_colum = torch.sum(uncorrupted, dim=0) # sum of each column
	highest_prob = list(torch.argsort(per_colum, descending=True).numpy())
	
	def gene_dist(tla_to_mod_to_kos, tla_to_kos, tla_to_tnum):
		gene_counts = []
		for org in tla_to_mod_to_kos:
			tnum = tla_to_tnum[org]
			if org in c_train_genomes and len(tla_to_mod_to_kos[org]) >= 10 and len(tla_to_kos[tnum]) >= 400:
				if len(tla_to_kos[tnum]) < 10:
					print()
					print (tnum, len(tla_to_kos[tnum]))
					print(len(tla_to_mod_to_kos[org]))
				gene_counts.append(len(tla_to_kos[tnum]))
		return gene_counts
	
	gene_counts = gene_dist(tla_to_mod_to_kos, tla_to_kos, tla_to_tnum)
	
	baseline2 = torch.zeros_like(uncorrupted)
	
	for i in range(baseline2.shape[0]):
		# determine how many genes we want to be on
		n_on = random.choice(gene_counts) 
		# how many are already on?
		already_on = [int(s) for s in (corrupted_test[i,:n_features] == 1).nonzero()]
		# remove already_on indices from highest_prob list, since they can't be turned on twice
		for s in already_on:
			if s in highest_prob:
				del highest_prob[highest_prob.index(s)]
		
		# get indices of the top n_on genes most likely to be "on" across all genomes
		idx_on = highest_prob[:int(n_on - len(already_on))]
		
		# merge new on and already on
		new = idx_on + already_on
		
		# turn on bits that should be on
		baseline2[i,:][new] = 1

	return baseline2.long()

def baseline4(corrupted_train, corrupted_test, tla_to_tnum, tnum_to_kos, c_train_genomes):
	"""
	Create baseline4 predictions: all generated genome vectors are just copies of the smallest genome vector in the training set (Hoaglandella endobia -- hed)
	
	Arguments:
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		corrupted_test (tensor) -- corrupted test data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		tla_to_tnum (dict) -- for each genome, converts tla to tnum
		tnum_to_kos (dict) -- maps tnums to KOs encoded by that genome, e.g.: 'T00001': [K00001, ... 'K0000N']
		c_train_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
	
	Returns:
		baseline4 (tensor) -- baseline4 predictions. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
	"""
	n_features = int(corrupted_train.shape[1]/2)
	
	### Find smallest genome in train set
	unique_train = list(set(c_train_genomes))
	tla_size = []
	for tla in unique_train:
		tnum = tla_to_tnum[tla]
		tla_size.append([tla, len(tla_to_kos[tnum])])
	sorted_tla_size = sorted(tla_size, key=lambda x: x[1], reverse=False)
	smallest_tla = sorted_tla_size[0][0] # tla = 'hed'
	print("smallest_tla",smallest_tla)
	# row index of smallest genome in train set
	start = c_train_genomes.index(smallest_tla) # hed = Hoaglandella endobia, Gammaproteobacteria
	smallest_uncorrupted = corrupted_train[start,n_features:]

	# Create baseline for test set
	baseline4 = torch.Tensor(np.tile(smallest_uncorrupted, (corrupted_test.shape[0], 1)))
	
	return baseline4.long()

def baseline5(corrupted_train, corrupted_test, tla_to_tnum, tnum_to_kos, c_train_genomes):
	"""
	Create baseline5 predictions: all generated genome vectors are just copies of the largest genome vector in the training set (_Paraburkholderia caribensis_ -- bcai)
	
	Arguments:
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		corrupted_test (tensor) -- corrupted test data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		tla_to_tnum (dict) -- for each genome, converts tla to tnum
		tnum_to_kos (dict) -- maps tnums to KOs encoded by that genome, e.g.: 'T00001': [K00001, ... 'K0000N']
		c_train_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
	
	Returns:
		baseline5 (tensor) -- baseline5 predictions. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
	"""	
	n_features = int(corrupted_train.shape[1]/2)
	
	### Find smallest genome in train set
	unique_train = list(set(c_train_genomes))
	tla_size = []
	for tla in unique_train:
		tnum = tla_to_tnum[tla]
		tla_size.append([tla, len(tla_to_kos[tnum])])
	sorted_tla_size = sorted(tla_size, key=lambda x: x[1], reverse=True)
	largest_tla = sorted_tla_size[0][0] # tla = hed
	print("largest_tla",largest_tla)	# row index of smallest genome in train set
	start = c_train_genomes.index(largest_tla) # hed = Hoaglandella endobia, Gammaproteobacteria
	largest_uncorrupted = corrupted_train[start,n_features:]

	# Create baseline for test set
	baseline5 = torch.Tensor(np.tile(largest_uncorrupted, (corrupted_test.shape[0], 1)))
	
	return baseline5.long(), largest_tla

def compare_in_n_out(binary_pred, corrupted):
	"""
	Plot histogram showing how often genes in the VAE input are also in the reconstruction / output
	
	Arguments:
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
		corrupted (tensor) -- corrupted data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		
	Returns:
		matplotlib.Figure
	"""
	out = {}
	for i, pred_row in enumerate(binary_pred):
		# what KOs were input?
		corrupt_row = corrupted[i,:]
		num_in = int(torch.sum(corrupt_row))
		# How many of the genes in the input are also in the output
		num_out = int(torch.sum(torch.mul(corrupt_row, pred_row)))
		out[i] = [num_out, num_in, float(num_out/num_in*100)]

	perc_out = [int(out[i][2]) for i in out]
	
	fig = fig, ax = plt.subplots()
	plt.hist(perc_out, bins=50)
	plt.xlabel('Percent of input genes in output')
	plt.ylabel('Count')
		
	count_hund = 0
	count_ninety = 0
	for i in perc_out:
		if i == 100: count_hund += 1
		if i >= 90: count_ninety += 1
	total = len(perc_out)
	
	print("There are",count_hund,"instance of inputs being 100% present in output")
	print("This is out of",total,"instances or",str(round(count_hund/total*100, 2))+"% of cases")
	print("There are",count_ninety,"instance of inputs being >=90% present in output ("+str(round(count_ninety/total*100, 2))+"%)")	
	return fig

def best_med_worst(f1s, c_test_genomes, tla_to_tnum):
	"""
	Get the best, median, and worst reconstructions from the test set, as measured by F1 score
	
	Arguments:
		f1s (list) -- test F1 scores
		c_test_genomes (list) -- tlas of genomes in the test set
		tla_to_tnum (dict) -- maps tla to tnum for each genome
		
	Returns:
		best (list) -- for the best reconstruction: index, tla, F1 score, tnum 
		median (list) -- for the median reconstruction: index, tla, F1 score, tnum
		worst (list) -- for the worst reconstruction: index, tla, F1 score, tnum
	"""
	idx_best = f1s.index(max(f1s))
	tla_best = c_test_genomes[idx_best]
	best = [idx_best, tla_best, f1s[idx_best], tla_to_tnum[tla_best]]
	
	# Get index of median F1 score
	f1s_sorted = sorted(f1s, reverse=True)
	idx_median = f1s.index(f1s_sorted[int(len(f1s_sorted)/2)])
	tla_median = c_test_genomes[idx_median]
	median = [idx_median, tla_median, f1s[idx_median], tla_to_tnum[tla_median]]
	
	idx_worst = f1s.index(min(f1s))
	tla_worst = c_test_genomes[idx_worst]
	worst = [idx_worst, tla_worst, f1s[idx_worst], tla_to_tnum[tla_worst]]
	
	return best, median, worst

def test_f1s(uncorrupted, binary_pred):
	"""
	Calculate F1 scores for all genomes in the test set and plot a histogram
	
	Arguments:
		uncorrupted (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
		
	Returns:
		f1s (list) -- F1 scores for each genome in the test set
		matplotlib.Figure
	"""
	f1s = []
	for i in range(0,len(binary_pred)):
		f1 = sk.metrics.f1_score(uncorrupted[i], binary_pred[i], zero_division=0)
		f1s.append(f1)
	
	print("median F1 score:",np.median(f1s))
	print("min F1 score", min(f1s))
	print("max F1 score", max(f1s))
	
	fig = fig, ax = plt.subplots()	
	plt.hist(f1s)
	plt.xlabel('F1 score')
	plt.ylabel('Count')
	
	return f1s, fig

def f1s_per_phylum(train_tax_dict, test_tax_dict, c_test_genomes, f1s):
	"""
	For training set, creates a dict counting the number of genome vectors (values) per phylum (keys). For test set, creates a dict of phyla (keys) + list of F1 scores of all genome vectors in that phylum (values).
	
	Arguments:
		train_tax_dict (dict) -- maps tla to [domain, phylum, ..., species] for all training set genomes
		test_tax_dict (dict) -- maps tla to [domain, phylum, ..., species] for all test set genomes
		c_test_genomes (list) -- -- tnum corresponding to each row (genome) of corrupted_test
		f1s (list) -- test F1 scores
		
	Returns:
		train_phyla (dict) -- keys are phylum names, values are the count of genome vectors per phylum in the training set
		test_phyla (dict of lists) -- keys are phylum names, values are lists of test set F1 scores per phylum
	"""

	test_phyla = {}
	for tla in test_tax_dict:
		phylum = test_tax_dict[tla][1]
		if phylum == "Proteobacteria":
			phylum = test_tax_dict[tla][2]
		if phylum not in test_phyla:
			test_phyla[phylum] = []
	
	train_phyla = {}
	for tla in train_tax_dict:
		phylum = train_tax_dict[tla][1]
		if phylum == "Proteobacteria":
			phylum = train_tax_dict[tla][2]
		if phylum not in train_phyla:
			train_phyla[phylum] = 1
		else:
			train_phyla[phylum] += 1
	
	for f1 in f1s:
		idx = f1s.index(f1)
		tla = c_test_genomes[idx]
		phylum = test_tax_dict[tla][1]
		if phylum == "Proteobacteria":
			phylum = test_tax_dict[tla][2]
		test_phyla[phylum].append(f1)

	return train_phyla, test_phyla


def plot_f1_per_phylum(test_phyla, figsize):
	"""
	Create barplot showing median F1 score per phylum in the test set
	
	Arguments:
		test_phyla (dict of lists) -- for each phylum in the test set (keys), list of F1 scores (float) for all genome vectors in that phylum
		figsize (tuple) -- figure size in (inches, inches)
	
	Returns:
		matplotlib.Figure
	"""
	mad = []
	median = []
	phylum_list = []
	for i in test_phyla:
		mad.append(stats.median_absolute_deviation(test_phyla[i]))
		median.append(np.median(test_phyla[i]))
		phylum_list.append(i)
		
	median, mad, phylum_list = zip(*sorted(zip(median, mad, phylum_list), reverse=True))
	phylum_num = [i for i in range(len(phylum_list))]
	
	plt.rcParams.update({'font.size': 12})
	fig = fig, ax = plt.subplots(figsize=figsize)  
	#plt.barh(phylum_num, median, yerr=mad)
	plt.barh(phylum_list, median, xerr=mad)
	plt.xlabel('Median F1 score')
	#plt.ylabel('Phylum')
	plt.ylim(-0.4,len(phylum_list)-0.6)
	print("Best:",phylum_list[0], median[0])
	print("Worst:", phylum_list[-1], median[-1])
	
	return fig

def plot_count_vs_f1s(train_phyla, test_phyla):

	phylum_f1s = [np.median(test_phyla[i]) for i in test_phyla]
	phylum_count = [train_phyla[i] for i in test_phyla]
	
	fig = fig, ax = plt.subplots() 
	plt.scatter(phylum_count, phylum_f1s)
	plt.xlabel("Number of genomes in train set")
	plt.ylabel("F1 score on test set")
	plt.xscale('log')
	
	return fig

def ngenesUncorrupted_vs_f1(uncorrupted_test, f1s, ax=None):
	"""
	Plots scatterplot showing # genes in uncorrupted training genomes vs F1 score of genes
	
	Arguments:
		uncorrupted_test (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome 
		f1s (list) -- test F1 scores
		ax (matplotlib.Axes) -- Axes to put figure in. If None, a new figure will be created.
		
	Returns:
		matplotlib.Figure
	"""
	n_genes_uncorrupted = torch.sum(uncorrupted_test, 1).numpy().tolist() # get sum of each row
	
	if ax is None:
		fig = plt.figure()
		ax = fig.get_axes()[0]
	else:
		fig = None
	ax.scatter(n_genes_uncorrupted, f1s, marker='.', s = 1)
	ax.set_xlabel("# genes in uncorrupted genome")
	ax.set_ylabel("F1 score")
	return fig

def ngenesCorrupted_vs_f1(corrupted_test, f1s):
	"""
	Plots scatterplot showing # genes in corrupted training genomes vs F1 score of genes
	
	Arguments:
		corrupted_test (tensor) -- corrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome 
		f1s (list) -- test F1 scores
		
	Returns:
		matplotlib.Figure	
	"""
	n_genes_corrupted = torch.sum(corrupted_test, 1).numpy().tolist() # get sum of each row
	
	fig = plt.figure() 
	plt.scatter(n_genes_corrupted, f1s, marker='.', s = 1)
	plt.xlabel("# genes in corrupted input")
	plt.ylabel("F1 score")
	
	return fig

def plot_train_count_hist(train_input_mods):
	"""
	Plots histogram showing the # of times each mod is used in a corrupted genome during training
	
	Arguments:
		train_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_train_genomes)
	
	Returns:
		matplotlib.Figure
	"""
	train_out = defaultdict(int)
	for genome in train_input_mods:
		for mod in genome:
			train_out[mod] += 1
			
	fig = fig, ax = plt.subplots()  
	plt.hist(train_out.values())
	plt.xlabel('# times mods are used in a corrupted genome')
	plt.ylabel('Count')	
	
	return fig

def learningNroc_curve(train_losses, test_losses, train_f1s, test_f1s, target, y_probas):
	"""
	Plots two learning curves and an ROC curve -- i.e. a pretty figure for the manuscript
	
	Arguments:
		train_losses (list) -- training losses (KLD + BCE)
		test_losses (list) -- test losses (KLD + BCE)
		train_f1s (list) -- training F1	scores
		test_f1s (list) -- test F1	scores
		target (numpy.ndarray) -- uncorrupted genomes, rows are genomes and columns are genes 
		y_probas (numpy.ndarray) -- model predictions, rows are genomes and columns are genes 
		
	Returns:
		matplotlib.Figure
	"""
	plt.rcParams.update({'font.size': 16})
	
	x_losses = [*range(len(train_losses))]
	
	fig, axs = plt.subplots(1, 3, figsize=(15, 5))
	
	axs[0].set_title("Optimization Learning Curve")
	axs[1].set_title("Performance Learning Curve")
	
	axs[0].set_ylim(10**4,10**7)
	axs[1].set_ylim(0,1)
	
	axs[0].plot(x_losses, train_losses, marker='.', c='#3385ff', label='Training', markersize=5)
	axs[0].plot(x_losses, test_losses, marker='.', c='#ff6666', label='CV', markersize=5)
	
	axs[1].plot(x_losses, train_f1s, marker='.', c='#3385ff', label='Training', markersize=5)
	axs[1].plot(x_losses, test_f1s, marker='.', c='#ff6666', label='CV', markersize=5)
	
	axs[0].set_xlim(-5,x_losses[-1]+5)
	axs[1].set_xlim(-5,x_losses[-1]+5)
	
	axs[0].set_ylabel('Loss (KLD + BCE)')
	axs[0].semilogy()
	axs[1].set_ylabel('F1 score')
	
	axs[0].set_xlabel('Experience')
	axs[1].set_xlabel('Experience')
	
	axs[1].axhline(y=max(test_f1s), color='r', dashes=(1,1))
	print("max F1 score", max(test_f1s))
	
	axs[0].legend(loc="upper right")
	axs[1].legend(loc="lower right")
		
	####### Begin ROC/AUC calculations
	fpr = dict()
	tpr = dict()
	roc_auc = dict()	
	
	n_genomes = target.shape[0]
	n_genes = target.shape[1]
	
	# Calculate scores for each individual gene
	for i in range(n_genes):
		fpr[i], tpr[i], thresh = roc_curve(target[:, i], y_probas[:, i])
		
		if np.isnan(fpr[i]).any():
			continue
			
	# Calculate micro-average
	fpr_micro, tpr_micro, _ = roc_curve(target.ravel(), y_probas.ravel())
	roc_auc["micro"] = auc(fpr_micro, tpr_micro)

	n_examples = 100 # number of example genes to plot on ROC curve
	
	# get colours for plotting
	cm = plt.cm.get_cmap('brg')
	c = np.linspace(0, 1, 50) # start, stop, how_many
	colours = [cm(i) for i in c]
	colours = colours*2
	
	# plot
	   
	ax = axs[2]
	a = random.sample(range(target.shape[1]), n_examples)
	for i in range(len(a)):
		plt.plot(fpr[a[i]], tpr[a[i]], color=colours[i], alpha=0.5,
			 lw=1) #, label=cluster_names[i]+" (AUC = %0.2f)" % roc_auc[i])
	plt.plot(fpr_micro, tpr_micro, color='black', 
			 lw=2, label='Micro-average (AUC = %0.2f)' % roc_auc["micro"])
	plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Micro-average')

	plt.xlim([-0.01, 1.01])
	plt.ylim([0, 1.0])

	axs[2].set_xlabel('False Positive Rate')
	axs[2].set_ylabel('True Positive Rate')
	axs[2].set_title('ROC Curve')
	
	plt.tight_layout()
	
	return fig

def genus_boxplot_stats(groups):
	"""
	Perform anova and tukey test to accompany analysis in genus_boxplot function
	
	Arguments:
		groups (list of lists) -- list1 = group1 = F1 scores for test genomes where one genome vector from the same genus was in the training set, etc.	
		
	Returns:
		an -- anova comparison results
		m_comp -- tukey test results
	"""
	scores = []
	dep_var = []
	
	for i, name in enumerate(groups):
	    scores.extend(name)
	    dep_var.extend([i]*len(name))
	df = pd.DataFrame([dep_var, scores], ['group', 'F1']).T
	
	# one way anova
	an = anova(data=df, dv='F1', between='group')
	
	m_comp = pairwise_tukeyhsd(endog=df['F1'], groups=df['group'], alpha=0.05)
	
	return an, m_comp

def plot_mod_count_vs_f1(test_input_mods, f1s):
	"""
	Create scatterplot of number of mod occurences in the corrupted training dataset vs F1 score on the test set
	
	Arguments:
		test_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_test_genomes)
		f1s (list) -- test F1 scores
		
	Returns:
		matplotlib.Figure
	"""
	# For each mod, for each time it occurs in a genome, append F1 score of genome reconstruction
	out = defaultdict(lambda: [])
	for idx,i in enumerate(test_input_mods):
		corruption_f1 = f1s[idx]
		for mod in i:
			out[mod].append(corruption_f1)
	
	mod_f1s = []
	mod_count = []
	for i in out:
		mod_f1s.append(np.median(out[i]))
		mod_count.append(len(out[i]))	 
	
	fig = fig, ax = plt.subplots() 
	plt.scatter(mod_count, mod_f1s)
	plt.xlabel("Number of mod occurences in the corrupted training dataset")
	plt.ylabel("F1 score on test set")
	plt.xscale('log')
	
def map_proc_mod():
	"""
	Map processes to modules and modules to process
	
	Returns:
		process_to_mod (dict) -- lists modules (values) within a process (keys)
		mod_to_proc (dict) -- lists process (values) within a modules (keys)
	"""
	process_to_mod = {}
	path = config.KEGG_MODS
	file = open(path).readlines()
	file = list(map(str.strip, file))
	
	type_proc = ""
	for s in file:
		if s[0] == "B" and len(s) > 5:
			type_proc = s.split(">")[1].split("<")[0]
	
		elif s[0] == "D": 
			mod = s.split()[1]
			if type_proc in process_to_mod:
				process_to_mod[type_proc].append(mod)
			else:
				process_to_mod[type_proc] = [mod]	
	
	mod_to_proc = {}
	for proc in process_to_mod:
		for mod in process_to_mod[proc]:
			mod_to_proc[mod] = proc
	
	return process_to_mod, mod_to_proc

def map_subproc_mod():
	"""
	Map subprocesses to modules and modules to subprocess
	
	Returns:
		subprocess_to_mod (dict) -- lists modules (values) within a subprocess (keys)
		mod_to_subproc (dict) -- lists subprocess (values) within a modules (keys)
	"""
	subprocess_to_mod = {}
	path = config.KEGG_MODS
	file = open(path).readlines()
	file = list(map(str.strip, file))
	
	type_proc = ""
	for s in file:
		if s[0] == "C" and len(s) > 5:
			type_proc = ' '.join(s.split()[1:])
	
		elif s[0] == "D": 
			mod = s.split()[1]
			if type_proc in subprocess_to_mod:
				subprocess_to_mod[type_proc].append(mod)
			else:
				subprocess_to_mod[type_proc] = [mod]	
	
	mod_to_subproc = {}
	for proc in subprocess_to_mod:
		for mod in subprocess_to_mod[proc]:
			mod_to_subproc[mod] = proc
	
	return subprocess_to_mod, mod_to_subproc

def plot_metab_pathway_f1_horizontal(process_to_mod, mod_to_kos_clean, all_kos, ko_f1s, figsize):
	"""
	Generate box plot showing F1 scores of genes within processes or subprocesses
	
	Arguments:
		process_to_mod
		mod_to_kos_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		all_kos (list) -- list of all KOs in the dataset
		figsize (tuple) -- (inches, inches)
		
	Returns:
		matplotlib.Figure
		proc_to_ko_F1s (dict) -- maps processes to kos to F1 scores
	"""
	proc_to_ko_F1s = defaultdict(list)
	for proc in process_to_mod:
		for mod in process_to_mod[proc]:
			try:
				kos = mod_to_kos_clean[mod]
				for ko in kos:
					idx = all_kos.index(ko)
					f1 = ko_f1s[idx]
					proc_to_ko_F1s[proc].append(f1)
			except KeyError: pass
	list_f1s = []
	list_procs = []
	list_medians = []
	
	for key in proc_to_ko_F1s:
		list_f1s.append(proc_to_ko_F1s[key])
		list_procs.append(key)
		list_medians.append(np.median(proc_to_ko_F1s[key]))
		
	list_medians, list_f1s, list_procs = zip(*sorted(zip(list_medians, list_f1s, list_procs), reverse=False))
	
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0,0,1,1])
	
	for i, proc in enumerate(list_procs):
		# add scatter on x-axis
		y = np.random.normal(i+1, 0.04, size=len(list_f1s[i]))
		plt.plot(list_f1s[i], y, 'r.', alpha=0.2)
		
	bp = ax.boxplot(list_f1s, showfliers=False, vert=False)
	
	plt.yticks([i+1 for i in range(len(list_procs))], [proc for proc in list_procs], rotation=0)
	plt.xlabel('F1 score')
	
	return fig, proc_to_ko_F1s

def export_selected_generated(BASE_DIR, gen_kos, gen_idx):
	"""
	Export a particular generated genome from our set of n used for the paper analysis
	
	Arguments:
		BASE_DIR (str) -- path to working dir
		gen_kos (list) -- KO numbers encoded by genome vector
		gen_idx (int) -- index of genome vector of interest
	"""
	date = pre_process.datenow()
	save_to = BASE_DIR+'prot_out_'+str(gen_idx)+'_'+date+'.txt'
	print('saving file to',save_to = BASE_DIR+'prot_out_'+str(gen_idx)+'_'+date+'.txt')
	
	with open(BASE_DIR+'seq_dict.pkl', 'rb') as handle:
		seq_dict = pickle.load(handle)
	
	with open(save_to, 'w') as handle:
		for prot in gen_kos:
			handle.write(">"+prot+"\n")
			handle.write(seq_dict[prot]+"\n")

def new_genome_random(mod_to_ko_clean, model, all_kos, BASE_DIR):
	"""
	Use DeepGenome to generate a new genome vector
	
	Arguments:
		mod_to_ko_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		model (genome_embeddings.models.VariationalAutoEncoder) -- trained VAE model
		all_kos (list) -- list of all KOs in the dataset
		BASE_DIR (str) -- path to working dir
	"""
	with open(BASE_DIR+'seq_dict.pkl', 'rb') as handle:
		seq_dict = pickle.load(handle)
	
	my_corrupted = torch.zeros(len(all_kos))
	
	# Pick 10 random modules as input
	n_mods = 10
	keeps = random.sample(list(mod_to_kos_clean.keys()), n_mods)
	
	# Get the genes for those modules
	idxs = []
	for mod in keeps:
		for ko in mod_to_kos_clean[mod]:
			idxs.append(all_kos.index(ko))
	
	my_corrupted[idxs] = 1
	
	# Make a predicted genome
	model.eval()
	with torch.no_grad():
		my_pred = model.forward(my_corrupted)[0].detach()
		
	my_binary_pred = eval_binarize(my_pred.reshape(1, -1), 0.5)
	
	# get indices that are turned on in the prediction
	on_idx = [i[1] for i in (my_binary_pred == 1).nonzero().tolist()]
	
	ko_ids = []
	for idx in on_idx:
		ko_ids.append(all_kos[idx])
		
	with open(save_to, 'w') as handle:
		for prot in ko_ids:
			handle.write(">"+prot+"\n")
			handle.write(seq_dict[prot]+"\n")
	
	return ko_ids

def generate_genomes(n_gen, all_kos, mod_to_kos, n_mods, model):
	"""
	Generate new genomes using a trained VAE model
	
	Arguments:
		n_gen (int) -- number of genomes to generate
		n_mods (int) -- number of modules to use as input
	
	Returns:
		generated (tensor) -- generated genome vectors. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		generated_inputs (dict) -- for each genome index, a list of lists. The first list is the modules that were used as inputs to the VAE, the second is the list of KOs that encode those modules
	"""
	generated = torch.zeros(n_gen, len(all_kos))
	generated_inputs = {} # track which modules were used as inputs for each generated genome
	
	for i in range(n_gen):
	
		my_corrupted = torch.zeros(len(all_kos))
	
		# Pick 10 random modules as input
		keeps = random.sample(list(mod_to_kos.keys()), n_mods)
	
		# Get the genes for those modules
		idxs = []
		kos = []
		for mod in keeps:
			for ko in mod_to_kos[mod]:
				idxs.append(all_kos.index(ko))
				kos.append(ko)
		
		# Turn them on in my vector
		my_corrupted[idxs] = 1
		
		# Save this information for later
		generated_inputs[i] = [keeps, kos]
		
		# Make a predicted genome
		with torch.no_grad():
			my_pred = model.forward(my_corrupted)[0].detach()
	
		my_binary_pred = eval_binarize(my_pred.reshape(1, -1), 0.5)
	
		# get indices that are turned on in the prediction
		on_idx = [i[1] for i in (my_binary_pred == 1).nonzero().tolist()]
		my_corrupted[on_idx] = 1
	
		generated[i] = my_corrupted

	return generated, generated_inputs

def pca_gen_vs_real(generated, test_data, idx=None):
	"""
	Plot PCA of Jaccard similarity between genomes, using Hamming distances as a metric
	
	Arguments:
		generated (tensor) -- generated genome vectors. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		test_data (numpy.ndarray) -- rows are genomes, columns are genes/KOs. 1's denote presence of a gene in the genome, 0's denote absence
		idx (int) -- highlight one genome vector, defined using their index, in red
	
	Returns:
		matplotlib.Figure
	"""	
	n_gen = generated.shape[0]
	
	# concatenate real and fake genomes
	concated = torch.cat((torch.Tensor(test_data), generated), 0).numpy()
	
	# generate labels
	test_data_labels = ['test' for i in range(test_data.shape[0])]
	generated_labels = ['generated' for i in range(n_gen)]
	
	# convert to df
	df = pd.DataFrame(concated)
	
	# calculate Jaccard similarity using Hamming distance metric
	jac_sim = 1 - pairwise_distances(df, metric = "hamming")
	
	# Do PCA
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(jac_sim)
	
	# Format things
	principalDf = pd.DataFrame(data = principalComponents
				 , columns = ['principal component 1', 'principal component 2'])
				 
	labels = test_data_labels + generated_labels
	#labels = test_data_labels + generated_labels + train_data_labels
	
	labels_df = pd.Series( (v for v in labels))
	
	finalDf = pd.concat([principalDf, labels_df], axis = 1)
	
	var_one = pca.explained_variance_[0]
	var_two = pca.explained_variance_[1]
	
	# Plot figure
	plt.rcParams.update({'font.size': 12})
	fig = plt.figure(figsize = (7.5,3))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 11)
	ax.set_ylabel('Principal Component 2', fontsize = 11)
	ax.grid()
	targets = ['test', 'generated'] #, 'train']
	colors = ['g', 'b']
	for target, color in zip(targets,colors):
		indicesToKeep = labels_df == target
		#print(finalDf.loc[indicesToKeep, 'principal component 1'])
		ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
				   , finalDf.loc[indicesToKeep, 'principal component 2']
				   , c = color
				   , s = 10)
	
	if idx:
		ax.scatter(finalDf.loc[[idx], 'principal component 1']
					   , finalDf.loc[[idx], 'principal component 2']
					   , c = 'r'
					   , s = 10)
	ax.legend(targets,loc='lower right')
	plt.axis('scaled')
  
	return fig
	
def df_for_phylip(generated, test_data, test_genomes, all_kos):
	"""
	Create df that will be useful in exporting data out to phylip
	
	Arguments:
		generated_genomes (tensor) -- generated genomes 
		test_data (np.ndarray) -- test genomes, rows = genomes, cols = genes
		test_genomes (list) -- genome IDs in same order as test_data 
		
	Returns:
		df (DataFrame) -- contains genome vectors for real (test) + generated genomes
	"""
	n_gen = generated.shape[0] # number of fake genomes
	
	# concatenate real and fake genomes
	concated = torch.cat((torch.Tensor(test_data), generated), 0).numpy()
	#concated = torch.cat((concated, torch.Tensor(train_data)), 0).numpy()
	
	generated_labels = ['generated'+str(i) for i in range(n_gen)]
	#train_data_labels = ['train' for i in range(train_data.shape[0])]
	labels = test_genomes + generated_labels
	df = pd.DataFrame(concated, index=labels, columns=all_kos)

	return df

def write_out_for_phylip(BASE_DIR, df, tnum_to_tla, test_tax_dict):
	"""
	Converts df of genome vectors into character matrix that can be input to Phylip, writes to disk
	
	Arguments:
		BASE_DIR (str) -- directory where file will be saved
		df (df) -- rows = genomes (ID is tnum), columns = genes
		tnum_to_tla (dict) -- converts tnum to tla (diff types of genome ID)
		test_tax_dict (dict of list) -- for each tla, list of domain, phylum, class, etc
		
	Returns:
		phylum_dict (dict) -- key = new ID for each genome, value = phylum 
		phylip_in.txt (txt saved to disk) -- character matrix for input to phylip
	"""
	date = pre_process.datenow() 
	save_to = BASE_DIR+"phylip_in"+date+".txt"
	print("file will be saved as",save_to)
	
	phylum_dict = {}
	with open(save_to, 'w') as handle:
		handle.write("	 "+str(df.shape[0])+"	 "+str(df.shape[1])+'\n')
		
		for idx, row in enumerate(df.iterrows()):
			index = str(idx)
			tnum = row[0]
			
			# get phylum for each real (test set) genome
			if tnum[0] == "T": # if it is a real, not generated, genome
				tla = tnum_to_tla[tnum]
				phylum = test_tax_dict[tla][1]
				if phylum == "Proteobacteria":
					phylum = test_tax_dict[tla][2]
			else:
				phylum = "Generated"
			
			# each genome will be renamed in such a way that phylip will accept the file
			# no illegal characters, same length names, etc
			if len(index) == 1:
				new_id = "s000"+index
			elif len(index) == 2:
				new_id = "s00"+index
			elif len(index) == 3:
				new_id = "s0"+index
			else: 
				new_id = "s"+index
			
			# save mapping of s0001 to "firmicutes" or "generated", etc
			phylum_dict[new_id] = phylum
			
			chars = df.iloc[idx].tolist()
			handle.write(new_id+"	 "+''.join([str(int(i)) for i in chars])+'\n')	
	
	return phylum_dict
	
def get_phyla_colours():
	"""
	Returns pre-defined dict mapping phyla in test set to a unique colour (rbg)
		
	Returns:
		phyla_colours (dict) -- keys = phylum name, values = colour in rbg format
	"""
	phyla_colours = {'Betaproteobacteria': (68, 226, 60),
	'Actinobacteria': (132, 192, 125),
	'Gammaproteobacteria': (158, 63, 170),
	'Alphaproteobacteria': (255,215,0),
	'Firmicutes': (18, 155, 192),
	'Tenericutes': (167, 198, 33),
	'Synergistetes': (54, 33, 115),
	'Deltaproteobacteria': (196, 60, 104),
	'Acidithiobacillia': (225, 179, 99),
	'Bacteroidetes': (119, 91, 189),
	'Thermotogae': (253, 74, 138),
	'Oligoflexia': (185, 105, 144),
	'Verrucomicrobia': (196, 71, 62),
	'Epsilonproteobacteria': (236, 126, 196),
	'Chloroflexi': (253, 214, 206),
	'Armatimonadetes': (188, 236, 114),
	'Chlorobi': (126, 106, 140),
	'Thermodesulfobacteria': (31, 29, 145),
	'Dictyoglomi': (214, 126, 140),
	'Elusimicrobia': (76, 246, 241),
	'Fusobacteria': (210, 187, 68),
	'Deferribacteres': (255,192,203),
	'Gemmatimonadetes': (122, 50, 183),
	'Ignavibacteriae': (140, 171, 221),
	'Planctomycetes': (252, 42, 9),
	'Nitrospirae': (117, 105, 45),
	'Zetaproteobacteria': (118, 53, 43),
	'Spirochaetes': (255,140,0),
	'Aquificae': (227, 50, 199),
	'Deinococcus-Thermus': (139, 200, 213),
	'Chlamydiae': (174, 160, 232),
	'Acidobacteria': (38, 15, 225),
	'Cyanobacteria': (142, 245, 178),
	'Candidatus Bipolaricaulota': (17, 105, 113),
	'Generated': (255, 255, 255)}
	
	return phyla_colours
	
def colour_real_itol(BASE_DIR, phyla_colours, phylum_dict):
	"""
	Creates iTOL colorstrip file for gene +/- dendrogram, colours = phyla of test genomes
	
	Arguments:
		BASE_DIR (str) -- directory in which to save file
		phyla_colours (dict) -- keys = phyla, values = unique colour (rbg)
		 phylum_dict (dict of lists) -- keys = phyla, values = list of tax (domain, phylum, class, etc)
	Returns:
		vae_dendro_colours_real.txt (saves to disk) -- colorstrip file
	"""
	date = pre_process.datenow() 
	save_to = BASE_DIR+"vae_dendro_colours_real"+date+".txt"
	print("file will be saved as", save_to)

	# legend shapes
	temp = [str(1) for i in range(len(phyla_colours))]
	legend_shapes = ' '.join(temp)

	# legends labels and colours 
	label_legend = []
	colour_legend = []
	for i in phyla_colours:
		name = i.replace(' ', '_')
		label_legend.append(name)
		colour_legend.append(phyla_colours[i])
	
	label_legend, colour_legend = zip(*sorted(zip(label_legend, colour_legend), reverse=False))
	
	legend_labels = ' '.join(label_legend)
	legend_colours = ''
	for i in colour_legend:
		legend_colours = legend_colours+' rgba('+str(i[0])+','+str(i[1])+','+str(i[2])+')'
	legend_colours = legend_colours[1:]
	
	with open(save_to, 'w') as handle:
		handle.write("DATASET_COLORSTRIP\n")
		handle.write("SEPARATOR SPACE\n")
		handle.write("DATASET_LABEL Phylum\n")
		handle.write("COLOR_BRANCHES 0\n")
		handle.write("LEGEND_TITLE Legend\n")
		handle.write("LEGEND_SHAPES "+legend_shapes+"\n")
		handle.write("LEGEND_COLORS "+legend_colours+"\n")
		handle.write("LEGEND_LABELS "+legend_labels+"\n")
		handle.write("MARGIN 5\n")
		handle.write("\n")
		handle.write("DATA\n")

		for i in phylum_dict:
			phylum = phylum_dict[i]
			if phylum not in phyla_colours:
				colour = tuple(np.random.randint(256, size=3))
				phyla_colours[phylum] = colour
			else:
				colour = phyla_colours[phylum]

			out = i+" rgba("+str(colour[0])+","+str(colour[1])+","+str(colour[2])+")\n"
			handle.write(out)

def colour_generated_itol(BASE_DIR, phylum_dict):
	"""
	Creates iTOL colorstrip file for gene +/- dendrogram, colours = real vs generated genome
	
	Arguments:
		BASE_DIR (str) -- directory in which to save file
		 phylum_dict (dict of lists) -- keys = phyla, values = list of tax (domain, phylum, class, etc)
	Returns:
		vae_dendro_colours_generated.txt (saves to disk) -- colorstrip file
	"""
	date = pre_process.datenow() 
	save_to = BASE_DIR+"vae_dendro_colours_generated"+date+".txt"
	print("file will be saved as",save_to)
	
	# legend shapes
	legend_shapes = '1 1'
	
	# legends labels
	legend_labels = 'Generated Real'
	
	# legend colours 
	legend_colours = 'rgba(0,0,0) rgba(255,255,255)'
	
	with open(save_to, 'w') as handle:
		handle.write("DATASET_COLORSTRIP\n")
		handle.write("SEPARATOR SPACE\n")
		handle.write("DATASET_LABEL Phylum\n")
		handle.write("COLOR_BRANCHES 0\n")
		handle.write("LEGEND_TITLE Legend\n")
		handle.write("LEGEND_SHAPES "+legend_shapes+"\n")
		handle.write("LEGEND_COLORS "+legend_colours+"\n")
		handle.write("LEGEND_LABELS "+legend_labels+"\n")
		handle.write("MARGIN 5\n")
		handle.write("\n")
		handle.write("DATA\n")

		phyla_done = {"Generated": 'rgba(0,0,0)', 'Real': 'rgba(255,255,255)'}
		for i in phylum_dict:
			phylum = phylum_dict[i]
			
			if phylum == "Generated":
				colour = phyla_done[phylum]
			else:
				colour = phyla_done['Real']
				
			out = i+" "+colour+"\n"
			handle.write(out)

def bio_insights_fig(test_phyla, subprocess_to_mod, all_kos, ko_f1s, mod_to_kos_clean):
	"""
	Generate two-panel figure; Panel 1 shows the median F1 score for test genomes from different phyla, Panel 2 shows the median F1 score of genes within different subprocesses
	
	Arguments:
		test_phyla (dict of lists) -- keys are phylum names, values are lists of test set F1 scores per phylum
		subprocess_to_mod (dict) -- lists modules (values) within a subprocess (keys)
		all_kos (list) -- list of all KOs in the dataset
		ko_f1s (list) -- F1 score of every KO, in the same order as they occur in uncorrupted_test
		mod_to_kos_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		
	Returns:
		matplotlib.Figure
	"""
	# Get data to plot for phylum analysis
	mad = []
	median = []
	phylum_list = []
	for i in test_phyla:
		mad.append(stats.median_absolute_deviation(test_phyla[i]))
		median.append(np.median(test_phyla[i]))
		phylum_list.append(i)
	
	median, mad, phylum_list = zip(*sorted(zip(median, mad, phylum_list), reverse=False))
	phylum_num = [i for i in range(len(phylum_list))]
	
	# Get data to plot for pathway analysis
	proc_to_ko_F1s = defaultdict(list)
	for proc in subprocess_to_mod:
		for mod in subprocess_to_mod[proc]:
			try:
				kos = mod_to_kos_clean[mod]
				for ko in kos:
					idx = all_kos.index(ko)
					f1 = ko_f1s[idx]
					proc_to_ko_F1s[proc].append(f1)
			except KeyError: pass
	list_f1s = []
	list_procs = []
	list_medians = []
	for key in proc_to_ko_F1s:
		list_f1s.append(proc_to_ko_F1s[key])
		list_procs.append(key)
		list_medians.append(np.median(proc_to_ko_F1s[key]))
	
	list_medians, list_f1s, list_procs = zip(*sorted(zip(list_medians, list_f1s, list_procs), reverse=False))
	
	# Create figure with subplots
	fig, [ax1, ax2] = plt.subplots(1, 2, sharey=False, figsize=(15, 10))
	plt.rcParams.update({'font.size': 12})
	
	ax1.barh(phylum_list, median, xerr=mad)
	ax1.set_xlabel('Median F1 score')
	ax1.set_xlim(0,1)
	ax1.set_ylim(-0.4,len(phylum_list)-0.6)
	
	for i, proc in enumerate(list_procs):
		# add scatter on x-axis
		y = np.random.normal(i+1, 0.04, size=len(list_f1s[i]))
		ax2.plot(list_f1s[i], y, 'r.', alpha=0.2)
	
	bp = ax2.boxplot(list_f1s, showfliers=False, vert=False)
	ax2.set_yticks([i+1 for i in range(len(list_procs))])
	ax2.set_yticklabels(list_procs) 
	ax2.set_xlabel('Median F1 score')
	ax2.set_xlim(0,1)
	
	plt.tight_layout()
	
	return fig

def confusion_barplot(f1s, c_test_genomes, tns, fps, fns, tps, uncorrupted, corrupted, idx):
	"""
	Create barplot showing # TNs, TPs, FNs, and FPs for original, corrupted, and reconstructed test genomes
	
	Arguments:
		f1s (list) -- test F1 scores
		c_test_genomes (list) -- -- tnum corresponding to each row (genome) of corrupted_test
		tns (list) -- number of true negatives for each genome
		fps (list) -- number of false positives for each genome
		fns (list) -- number of false negatives for each genome
		tps (list) -- number of true positives for each genome
		uncorrupted (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		corrupted (tensor) -- corrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		idx (int) -- genome index in c_test_genomes
		
	Returns:
		matplotlib.Figure
	"""
	num_charts = 3 # number of genomes for which to make pie charts
	indices = [i for i in range(len(f1s))] # original index of each genome in uncorrupted
	sorted_f1s, sorted_indices = zip(*sorted(zip(f1s, indices), reverse=True))
	
	# Many of the best reconstructions are from the same few genomes 
	# (e.g.: model does really well reconstructing babt)
	# Take best F1 scores for the top three different uncorrupted genomes
	keeps_idx = []
	seen = []
	
	for i, _ in enumerate(sorted_f1s):
		orig_idx = sorted_indices[i]
		tla = c_test_genomes[orig_idx]
		if tla in seen: pass
		else: 
			keeps_idx.append(orig_idx)
			seen.append(tla)
	
	fig, axs = plt.subplots(1, 1, figsize=(15,3))

	# GENERATED
	posn = 2 # which subplot
	orig_idx = idx
	f1_best = f1s[orig_idx]
	tn_best = tns[orig_idx]
	fp_best = fps[orig_idx]
	fn_best = fns[orig_idx]
	tp_best = tps[orig_idx]
	data1 = [tp_best, tn_best, fn_best, fp_best]
	total = sum(data1)
	perc1 = [round(tp_best/total*100,2), round(tn_best/total*100,2), round(fn_best/total*100,2), round(fp_best/total*100,2)]
	
	# CORRUPTED
	posn = 1
	tp_best = int(torch.sum(corrupted[orig_idx]))
	tn_best = int(corrupted.shape[1] - torch.sum(corrupted[orig_idx]))
	data2 = [tp_best, tn_best]	
	perc2 = [round(tp_best/total*100,2), round(tn_best/total*100,2), 0, 0]

	# UNCORRUPTED
	posn = 0
	tp_best = int(torch.sum(uncorrupted[orig_idx]))
	tn_best = int(uncorrupted.shape[1] - torch.sum(uncorrupted[orig_idx]))
	data3 = [tp_best, tn_best]
	perc3 = [round(tp_best/total*100,2), round(tn_best/total*100,2), 0, 0]

	N = 3
	r = range(N)
	
	bars1 = np.array([perc1[1], perc2[1], perc3[1]]) # tps
	bars2 = np.array([perc1[0], perc2[0], perc3[0]]) # tps
	bars3 = np.array([perc1[2], perc2[2], perc3[2]]) # tps
	bars4 = np.array([perc1[3], perc2[3], perc3[3]]) # tps
	
	# TN = black
	# TP = green
	# FN =  magenta
	# FP = yellow
	colors = ['black', 'green', 'magenta', 'yellow']
	labels = ["TN", "TP", "FN", "FP"]
	
	barWidth = 1
	lefts = 0
	for bars, col, label in zip([bars1, bars2, bars3, bars4], colors, labels):
		axs.barh(r, bars, left=lefts, color=col, edgecolor='white', height=barWidth, label=label)
		lefts += bars


	axs.legend()
	print("labels", labels)
	axs.set_xlim([0,100])
	axs.set_ylim(-0.5, len(bars) - 0.5)
	axs.title.set_text(c_test_genomes[orig_idx]+", F1: "+str(round(f1_best,2)))
	axs.set_yticklabels(['Generated', 'Corrupted', 'Original', ''])
	
	axs.set_xlabel('Percent (%)')
	
	print(c_test_genomes[orig_idx],"F1: "+str(f1_best))
	print("generated genome:",data1)
	print("generated genome:",perc1)
	print()
		
	plt.tight_layout()

	return fig 

def arch_root(all_kos):
	"""
	Get archaeal outgroup genome vector for building a dendrogram
	
	Arguments:
		all_kos (list) -- list of all KOs in the dataset
	
	Returns:
		barc_vec (list) -- archaeal genome vector
	"""
	path = config.ANNOTATIONS_PATH
	file = open(path+'barc_annotations.txt').readlines()
	file = list(map(str.strip, file))
	
	barc_kos = []
	for s in file:
		if "<a href=" in s:
			x = s.split()[2]
	#			 if "K00668" in s:
	#				 print("s",s)
	#				 print("x", x)
	#				 print()
			if re.match(r'[K]\d{5}', x): 
				barc_kos.append(x) #[K]\d{5}
	
	barc_vec = []
	for ko in all_kos:
		if ko in barc_kos:
			barc_vec.append(1)
		else:
			barc_vec.append(0)
	
	return barc_vec

def get_mod_names():
	"""
	Get names of all modules (e.g.: 'M00001': 'Glycolysis (Embden-Meyerhof pathway)')
	
	Returns:
		mod_names (dict) -- maps 5-letter name to full english name of all mods
	"""
	
	process_to_mod = {}
	path = config.KEGG_MODS
	file = open(path).readlines()
	file = list(map(str.strip, file))
	
	type_proc = ""
	for s in file:
		if s[0] == "D": 
			mod_names[s.split()[1]] = ' '.join(s.split()[2:]).split('[')[0].split(',')[0]
			
	return mod_names 
	
def compare_inputs(test_input_mods, idx, tla_to_mod_to_kos, train_genomes, tla_to_tnum, mod_names):
	"""
	For a given generated genome vector, figure out how many training set genome vectors encoded all 10 of the modules used as input to generate our selected genome vector. Create a barplot.
	
	Arguments:
		test_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_test_genomes)
		idx (int) -- index of generated genome vector in test_input_mods
		tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
		train_genomes (list) -- tnums of genomes in the training set
		tla_to_tnum (dict) -- for each genome, converts tla to tnum
		mod_names
		
	Returns:
		matplotlib.Figure
		all_ten (list) -- tnums of all train genomes that encode all 10 input modules used for selected generated genome vector	
	"""
	# What input modules were used for selected generated genome vector?
	gen_mods = test_input_mods[idx]	
	
	# Which orgs have those mods?
	mod_count = defaultdict(int)
	all_ten = []
	for tla in tla_to_mod_to_kos:
		try:
			tnum = tla_to_tnum[tla]
		except KeyError: pass
		if tnum not in train_genomes: continue
		mods = list(tla_to_mod_to_kos[tla].keys())
		for mod in mods:
			if mod in gen_mods:
				mod_count[mod] += 1 
		
		# of input mods to selected generated genome vector, how many genomes have all ten?
		all_present = True
		for i in gen_mods:
			if i not in mods: all_present = False
		if all_present:
			all_ten.append(tnum) 	
	
	
	mods = [mod_names[i] for i in mod_count.keys()]
	vals = mod_count.values()
	vals, mods = zip(*sorted(zip(vals, mods), reverse=False))
	
	fig, ax = plt.subplots()
	plt.barh(mods, vals, color='#3385ff')
	plt.xlabel("# genomes encoding module")
	
	return fig, all_ten
	
def compare_venn(name1, name2, name3, df):
	"""
	Make Venn diagram showing overlap in gene IDs between three genome vectors
	
	Arguments:
		name1 (str) -- species level name of one genome
		name2 (str) -- species level name of a second genome
		name3 (str) -- species level name of a third genome (e.g. "Generated")
		df (pd.DataFrame) -- column names are species, each row is a genome vector
		
	Returns:
		matplotlib.Figure
		matplotlib.axes
	"""
	genome1 = df.loc[name1].tolist()
	genome2 = df.loc[name2].tolist()
	genome3 = df.loc[name3].tolist()
	
	shared1 = []
	shared2 = []
	shared3 = []
	for i in range(len(genome1)):
		if genome1[i] == genome2[i] == genome3[i] == 1:
			shared1.append(str(i)+'shared123')
			shared2.append(str(i)+'shared123')
			shared3.append(str(i)+'shared123')
		elif genome1[i] == genome2[i] == 1:
			shared1.append(str(i)+'shared12')
			shared2.append(str(i)+'shared12')  
		elif genome1[i] == genome3[i] == 1:
			shared1.append(str(i)+'shared13')
			shared3.append(str(i)+'shared13')   
		elif genome2[i] == genome3[i] == 1:
			shared2.append(str(i)+'shared23')   
			shared3.append(str(i)+'shared23')
		elif genome1[i] == 1:
			shared1.append(str(i)+'unique')
		elif genome2[i] == 1:
			shared2.append(str(i)+'unique')
		elif genome3[i] == 1:
			shared3.append(str(i)+'unique')
	
	fig, ax = plt.subplots(figsize=(6, 6))
	fig = venn3([set(shared1), set(shared2), set(shared3)], 
	set_labels = (name1, name2, name3))
	
	return fig, ax

def get_ten_closest(index, tnum_x, test_genomes, train_genomes, uncorrupted_test, unc_train_data, binary_pred, train_tax_dict, test_tax_dict, tnum_to_tla):
	"""
	For a given reconstructed genome vector, get the ___ and ___ of the most similar real training genome vectors (similar is defined by the Hamming distance)
	
	Arguments:
		index (int) -- index in test ds of genome vector reconstruction of interest
		tnum (str) -- tnum of genome vector reconstruction of interest
		test_genomes (list) -- tnums of test genomes
		train_genomes (list) -- tnums of train genomes
		uncorrupted_test (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome 
		train_data (tensor) -- training data
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
		train_tax_dict (dict) -- maps tla to [domain, phylum, ..., species] for all training set genomes
		test_tax_dict (dict) -- maps tla to [domain, phylum, ..., species] for all test set genomes
		tnum_to_tla (dict) -- maps tnum to tla for every genome in the ds
		
	Returns:
		ten_df -- df of genome vectors of the top 10 closest genome vectors + original uncorrupted + generated
		closest_genomes (list) -- tnums of the top closest genome vectors in order of closest to least close
	"""
	# Get reconstruction genome vector 
	generated = binary_pred[index,:]
	# Get original, uncorrupted genome vector from which it was derived
	idx_v2 = test_genomes.index(tnum_x) # indexing is different for binary_pred and test_genomes
	orig = uncorrupted_test[idx_v2,:] # original 
	
	# Calculate hamming distances between reconstruction and training set original genomes
	hammings = []
	idxs = []
	for i, row in enumerate(unc_train_data):
		hl = hamming_loss(generated, row)
		hammings.append(hl)
		idxs.append(i)	
		
	# Find top 10 closest genome vectors
	hammings, train_genomes_sorted, idxs = zip(*sorted(zip(hammings, train_genomes, idxs), reverse=False))
	hamm_10 = hammings[:10]
	closest_genomes = train_genomes_sorted[:10] # tnums of top 10 closest genomes
	idx_10 = idxs[:10]
		
		
	# First get species-level names for each of the 10 closest training set genomes 
	# Get species-level name of the original input genome for VAE reconstruction 
	# Include label for the generated genome "Generated"
	labels = [train_tax_dict[tnum_to_tla[tnum]][6] for tnum in closest_genomes] \
					+ [test_tax_dict[tnum_to_tla[tnum_x]][6],'Generated']	
	
	ten_df = pd.DataFrame(np.vstack((unc_train_data[idx_10, :], orig, generated)), labels)
	
	return ten_df, closest_genomes
	
def make_pred(new_preds, model, corrupted, binarizer_threshold, name):
	"""
	Make predictions using model
	
	Arguments:
		new_preds (Bool) -- [True | False] Indicates whether to make new predictions or load ones from file
		model (genome_embeddings.models.VariationalAutoEncoder) -- trained VAE model
		corrupted (tensor) -- corrupted data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome
		binarizer_threshold -- probability threshold that dictates whether a pred should be a 1 or a 0
		name -- path + unique file name to save / load predications
		
	Returns:
		pred (tensor) -- for each genome in corrupted, y_probas prediction as to which genes should be on/off
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
	"""
	if new_preds:
		model.eval()
		with torch.no_grad():
			pred = model.forward(corrupted)[0].detach()
		binary_pred = eval_binarize(pred, binarizer_threshold)
		torch.save(pred, name+"_preds.pt")
		torch.save(binary_pred, name+"_binary_preds.pt")
	else:
		pred = torch.load(name+"_preds.pt")
		binary_pred = torch.load(name+"_binary_preds.pt")
		
	return pred, binary_pred

def nmods_vs_f1(c_test_genomes, test_input_mods, tla_to_mod_to_kos, tla_to_tnum, train_genomes, f1s, ax=None):
	"""
	Plots scatterplot showing correlation between F1 score of reconstruction using a given module as input and how many training set genomes encode that module
	
	Arguments:
		c_test_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
		test_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_test_genomes)
		tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
		tla_to_tnum (dict) -- for each genome, maps tla to tnum
		train_genomes (list) -- list of tnums in training set
		f1s (list) -- list of F1 scores for test set reconstructions
		ax (matplotlib.Axes) -- Axes to put figure in. If None, a new figure will be created
	
	Returns:
		matplotlib.Figure
		num_mods (list) -- for each mod, the number of genomes that encode it
	"""
	num_mods = []
	for i, tla in enumerate(c_test_genomes):
			
		# which mods were used as input to this genome
		gen_mods = test_input_mods[i]
		
		# Which orgs have those mods?
		all_ten = []
		for tla_compare in tla_to_mod_to_kos:
			try:
				tnum = tla_to_tnum[tla_compare]
			except KeyError: pass
			if tnum not in train_genomes: continue
			if tla == tla_compare: continue
				
			# of input mods, how many genomes have all ten?
			mods = list(tla_to_mod_to_kos[tla_compare].keys())
			all_present = True
			for s in gen_mods:
				if s not in mods: all_present = False
			if all_present:
				all_ten.append(tnum)
				
		num_mods.append(len(all_ten))
	
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = None
		
	ax.scatter(num_mods, f1s)
	
	return fig, num_mods		
	
def plot_tla_to_kos(c_test_genomes, tla_to_tnum, train_genomes, tnum_to_tax, tax_groups, f1s, ax=None):
	"""
	Barplot showing the # of same-genus genome vectors in the training set vs test set F1 scores
	
	Arguments:
		tnum_to_tax (dict of lists) -- maps tnum to taxonomy in form of [domain, phylum, ..., species]
		train_genomes (list) -- list of tnums in training set
		c_test_genomes (list) -- tnum corresponding to each row (genome) of corrupted_test
		tla_to_tnum (dict) -- for every genome, maps tla to tnum
		train_genomes (list) -- tnums of train genomes
		tax_groups (dict of lists) -- for each taxonomic level (key), list of taxa in that group (at that tax level)
		f1s (list) -- list of F1 scores for test set reconstructions
		ax (matplotlib.Axes) -- Axes to put figure in. If None, a new figure will
			be created.
		
	Returns:
		matplotlib.Figure
		groups (list of lists) -- list1 = group1 = F1 scores for test genomes where one genome vector from the same genus was in the training set, etc.	
	"""
	# Parse out how many genomes are assigned to each taxonomic level
	tax_groups = {'phylum': [],
				 'class': [],
				 'order': [],
				 'family': [],
				 'genus': [],
				 'species': []} # taxonomic groups in training set
	for tnum in tnum_to_tax:
		if tnum in train_genomes:
			tax_groups['phylum'].append(tnum_to_tax[tnum][1])
			tax_groups['class'].append(tnum_to_tax[tnum][2])
			tax_groups['order'].append(tnum_to_tax[tnum][3])
			tax_groups['family'].append(tnum_to_tax[tnum][4])
			tax_groups['genus'].append(tnum_to_tax[tnum][5])
			tax_groups['species'].append(tnum_to_tax[tnum][6])

	# Generate counts
	genus_count = defaultdict(int)
	genus_f1 = defaultdict(list)
	for i, tla in enumerate(c_test_genomes):
		tnum = tla_to_tnum[tla]
		genus = tnum_to_tax[tnum][5]
		genus_count[genus] = tax_groups['genus'].count(genus)
		genus_f1[genus].append(f1s[i])
	
	# Get median F1 for each genus
	group_0 = []
	group_1 = []
	group_2 = []
	group_3 = []
	group_4 = []
	group_5 = []
	genus_results_b = []
	for genus in genus_count:
		if genus_count[genus] > 6: continue # skip '' (unknown genus)
		if genus_count[genus] == 0: group_0.append(np.median(genus_f1[genus]))
		if genus_count[genus] == 1: group_1.append(np.median(genus_f1[genus]))
		if genus_count[genus] == 2: group_2.append(np.median(genus_f1[genus]))
		if genus_count[genus] == 3: group_3.append(np.median(genus_f1[genus]))
		if genus_count[genus] == 4: group_4.append(np.median(genus_f1[genus]))
		if genus_count[genus] == 5: group_5.append(np.median(genus_f1[genus]))
	
	# Let's actually make the figure
	if ax is None:
		fig, ax = plt.subplots(figsize=(6, 6))
	else:
		fig = None
	
	for i, group in enumerate([group_0, group_1, group_2, group_3, group_4, group_5]):
		x = np.random.normal(1+i, 0.08, size=len(group)) # scatter
		ax.plot(x, group, color='#1f77b4', marker='.', linestyle="None", alpha=0.5, markersize = 5) 

	bp = ax.boxplot([group_0, group_1, group_2, group_3, group_4, group_5], labels=[0,1,2,3,4,5], sym='.', showfliers=False) 
	
	# Change outline color, fill color and linewidth of the boxes
	for box in bp['boxes']:
		# Change outline color
		box.set( color='#000000', linewidth=0.6, linestyle='-')
	
	# Change linewidth of the whiskers
	for whisker in bp['whiskers']:
		whisker.set(color='#000000', linewidth=0.6, linestyle='-')
	
	# Change color and linewidth of the caps
	for cap in bp['caps']:
		cap.set(color='#000000', linewidth=0.6)
	
	# Change color and linewidth of the medians
	for median in bp['medians']:
		median.set(color='#000000', linewidth=0.6)
		
	ax.set_xlabel('# of same-genus genome vectors in training set')
	ax.set_ylabel('F1 score')
	
	groups = [group_0, group_1, group_2, group_3, group_4, group_5]
					   
	return fig, groups

def f1_per_ko(uncorrupted_test, binary_pred, train_data):
	"""
	Calculate the F1 score of every KO and generate a histogram
	
	Arguments:
		uncorrupted_test (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome 
		binary_pred (tensor) -- for each genome in corrupted, binary predications as to which genes should be on/off
		train_data (tensor) -- training data
		
	Returns:
		matplotlib.Figure
		ko_f1s (list) -- F1 score of every KO, in the same order as they occur in uncorrupted_test
	"""
	ko_f1s = []
	for i in range(uncorrupted_test.shape[1]): # for every column
		f1 = sk.metrics.f1_score(uncorrupted_test[:,i], binary_pred[:,i], zero_division=0)
		ko_f1s.append(f1)
		
	zeros = 0
	for i in ko_f1s:
		if i == 0:
			zeros += 1
	print("There are",zeros,"KOs with F1=0 out of",len(ko_f1s),"KOs. That is", zeros/len(ko_f1s)*100,"%")	
	
	# In the training set, how many KOs are always zero?
	zeros_train = train_data.sum(axis=0) > 0
	n_ones = np.sum(zeros_train)
	n_zeros_train = len(zeros_train) - n_ones
	print("There are",n_zeros_train,"genes that are always off in the training set")
	
	fig = plt.figure()
	plt.hist(ko_f1s, bins = 50)
	plt.xlabel("F1 score per gene")
	plt.ylabel("Count")
	
	return fig, ko_f1s

def geneCount_vs_geneF1(corrupted_train, num_features, ko_f1s, ax=None):
	"""
	Create scatter plot of gene count in the uncorrupted training set vs per gene test F1 score
	
	Arguments:
		corrupted_train (tensor) -- corrupted training data
		num_features (int) -- number of genes in the ds
		ko_f1s (list) -- F1 score of every KO, in the same order as they occur in uncorrupted_test
		ax (matplotlib.Axes) -- Axes to put figure in. If None, a new figure will
			be created.
	
	Returns:
		matplotlib.Figure
	"""

	tr_uncorrupted = corrupted_train[:,num_features:]
	ko_counts = torch.sum(tr_uncorrupted, 0)
	
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = None
		
	ax.scatter(ko_counts, ko_f1s, marker='.', s = 1)
	ax.set_xlim(0, tr_uncorrupted.shape[0])
	ax.set_ylim(0,1)
	ax.set_xlabel("gene count in uncorrupted train set")
	ax.set_ylabel("per gene test F1 score")
	plt.sca(ax)
	plt.xticks(rotation=-70)
#	ax.set_xticks(ax.get_xticks(), rotation=-70)
	print("max KO count:",int(max(ko_counts)))
	print("total number of training genomes:",tr_uncorrupted.shape[0])
	
	return fig

def model_performance_factors(c_test_genomes, tla_to_tnum, tnum_to_tax, tax_groups, f1s, corrupted_train, num_features, ko_f1s, uncorrupted_test, train_genomes, test_input_mods, tla_to_mod_to_kos):
	"""
	Arguments:
		c_test_genomes (list) -- -- tnum corresponding to each row (genome) of corrupted_test
		tla_to_tnum (dict) -- for each genome, converts tla to tnum
		tnum_to_tax (dict of lists) -- maps tnum to taxonomy in form of [domain, phylum, ..., species]
		tax_groups (dict of lists) -- for each taxonomic level (key), list of taxa in that group (at that tax level)
		f1s (list) -- test F1 scores
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		num_features (int) -- number of genes in dataset
		ko_f1s (list) -- F1 score of every KO, in the same order as they occur in uncorrupted_test
		uncorrupted_test (tensor) -- uncorrupted test data; rows = genomes, columns = genes; 1 = gene encoded by genome, 0 = absent from genome 
		train_genomes (list) -- list of tnums in training set
		test_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_test_genomes)
		tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
		
	"""
	fig, axs = plt.subplots(2,2, figsize=(10, 8))
	
	ax1 = axs[0,0]
	ax2 = axs[0,1]
	ax3 = axs[1,0]
	ax4 = axs[1,1]
	
	# Panel A
	plot_tla_to_kos(c_test_genomes, tla_to_tnum, train_genomes, tnum_to_tax, tax_groups, f1s, ax=ax1)
	
	# Panel B
	geneCount_vs_geneF1(corrupted_train, num_features, ko_f1s, ax=ax2)
	
	# Panel C
	ngenesUncorrupted_vs_f1(uncorrupted_test, f1s, ax=ax3)
	
	# Panel D
	nmods_vs_f1(c_test_genomes, test_input_mods, tla_to_mod_to_kos, tla_to_tnum, train_genomes, f1s, ax=ax4)
	
	plt.tight_layout()
	
	return fig
	
def complete_mods(generated, all_kos, mod_to_ko_clean):
	"""
	Calculate the number of complete modules (all req'd KOs are present) in a set of genomes
	
	Arguments:
		generated (tensor) -- generated genome vectors. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		all_kos (list) -- list of all KOs in the dataset
		mod_to_ko_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		
	Returns:
		gen_mods (defaultdict) -- for each genome vector (denoted by their index), list of complete mods encoded
	"""
	gen_kos = defaultdict(list)
	for i, row in enumerate(generated):
		for j in range(len(row)):
			if row[j] == 1:
				gen_kos[i].append(all_kos[j])

	gen_mods = defaultdict(list)
	for genome in gen_kos:
		my_kos = gen_kos[genome]

		for mod in mod_to_ko_clean:
			complete = True
			for ko in mod_to_ko_clean[mod]:
				if ko not in my_kos: 
					complete = False

			if complete:
				gen_mods[genome].append(mod)
				
	return gen_mods

def mod_freqs(mod_to_ko_clean, test_data, generated, real_mods, gen_mods):
	"""
	Calculate the frequency of modules in real test genomes and generated genomes
	
	Arguments:
		mod_to_ko_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		test_data (numpy.ndarray) -- rows are genomes, columns are genes/KOs. 1's denote presence of a gene in the genome, 0's denote absence
		generated (tensor) -- generated genome vectors. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		real_mods (defaultdict) -- for each real genome vector (denoted by their index), list of complete mods encoded
		gen_mods (defaultdict) -- for each generated genome vector (denoted by their index), list of complete mods encoded
		
	Returns:
		real_mod_freq (tuple) -- for each module, the fraction of real genomes that encode it
		gen_mod_freq (tuple) -- for each module, the fraction of generated genomes that encode it
	"""
	real_mod_freq = []
	gen_mod_freq = []
	for mod in mod_to_ko_clean:
		real_count = 0
		gen_count = 0
		for genome in gen_mods:
			if mod in gen_mods[genome]:
				gen_count += 1
			if mod in real_mods[genome]:
				real_count += 1
		real_mod_freq.append(real_count / len(test_data))
		gen_mod_freq.append(gen_count / len(generated))

	# sort in descending order of real genome mods
	real_mod_freq, gen_mod_freq = zip(*sorted(zip(real_mod_freq, gen_mod_freq), reverse=True))

	return real_mod_freq, gen_mod_freq

def dist_genes_mods(generated, all_kos, mod_to_ko_clean, test_data):
	"""
	Generates three panel figure. Panel 1 is a barplot of # of genes vs genome count, Panel #2 is a barplot of # of complete modules vs genome count, and Panel #3 is a barplot of module vs fractino of genomes encoding that module
	
	Arguments:
		generated (tensor) -- generated genome vectors. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		all_kos (list) -- list of all KOs in the dataset
		mod_to_ko_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		test_data (numpy.ndarray) -- rows are genomes, columns are genes/KOs. 1's denote presence of a gene in the genome, 0's denote absence
		
	Returns:
		matplotlib.Figure
	"""
	# First crunch some data
	gen_mods = complete_mods(generated, all_kos, mod_to_ko_clean)
	real_mods = complete_mods(test_data, all_kos, mod_to_ko_clean)
	gen_mod_lens = [len(gen_mods[i]) for i in gen_mods]
	real_mod_lens = [len(real_mods[i]) for i in gen_mods]
	
	real_mod_freq, gen_mod_freq = mod_freqs(mod_to_ko_clean, test_data, generated, real_mods, gen_mods)
	labels = [i for i in range(len(gen_mod_freq))]
	
	len_gen = []
	for genome in generated:
		len_gen.append(torch.sum(genome))
	len_real = []
	for genome in test_data:
		len_real.append(np.sum(genome))	
	
	# Plot a figure
	plt.rcParams.update({'font.size': 18})
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
	#plt.yticks(fontsize=20) 
	
	# Plot number of genes per genome
	ax1.hist(len_real, 50, color='g', alpha=0.5)
	ax1.hist(len_gen, 50, color='b', alpha=0.5)
	#ax1.legend(['Real', 'Generated'])
	ax1.set_xlabel("Number of genes")
	ax1.set_ylabel("Genome count")
	
	# Plot number of complete mods per genome
	ax2.hist(gen_mod_lens, 50, color='b', alpha=0.5)
	ax2.hist(real_mod_lens, 50, color='g', alpha=0.5)
	#ax2.legend(['Real', 'Generated'])
	ax2.set_xlabel("Number of complete modules")
	ax2.set_ylabel("Genome count")
	
	# Plot the fraction of genomes encoding each mod
	ax3.bar(labels, gen_mod_freq, color='b', alpha=0.5)
	ax3.bar(labels, real_mod_freq, color='g', alpha=0.5)
	ax3.legend(['Real', 'Generated'])
	ax3.set_xlabel("Module")
	ax3.set_ylabel("Fraction of genomes \n encoding module")
	ax3.set_xlim(0,len(labels))
	
	plt.tight_layout()
	
	return fig

def kos_in_gen(generated, gen_idx, all_kos):
	"""
	Return list genes/KOs encoded by a generated genome vector
	
	Arguments:
		generated (tensor) -- generated genome vectors. Rows are genomes, columns are genes. 1's denote a gene is encoded, 0 denotes that it is not
		gen_idx (int) -- index of generated genome in set of real + generated genomes
		all_kos (list) -- list of all KOs in the dataset
		
	Returns:
		gen_kos (list) -- KO numbers encoded by genome vector
	"""
	gen_ko_idx = [int(i) for i in (generated[gen_idx] == 1).nonzero()]
	gen_kos = [all_kos[i] for i in gen_ko_idx]
	print("There are a total of",len(gen_kos),"genes encoded in this genome vector")
	
	return gen_kos

def id_incomplete_mods(generated_inputs, gen_idx, mod_to_ko_clean, gen_kos):
	"""
	Identify incomplete modules in a generated genome vector, learn more about how they are incomplete.
	
	Arguments:
		generated_inputs (dict) -- for each genome index, a list of lists. The first list is the modules that were used as inputs to the VAE, the second is the list of KOs that encode those modules
		gen_idx (int) -- index of generated genome in set of real + generated genomes
		mod_to_ko_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		gen_kos (list) -- KO numbers encoded by genome vector
	"""
	# for each mod number, get its name
	mod_to_name = pre_process.mod_names()
	
	def mod_completeness(genome_vector, mod_to_ko_clean, mod):
		count = 0
		for i in mod_to_ko_clean[mod]:
			if i in genome_vector:
				count += 1
			else:
				print("missing", i)
		print(count,"/",len(mod_to_ko_clean[mod]),"genes in the mod are present")
		
	for mod in generated_inputs[gen_idx][0]:
		print(mod, mod_to_name[mod])
		print(mod_to_ko_clean[mod])
		mod_completeness(gen_kos, mod_to_ko_clean, mod)
		print("--------------------------------------------")