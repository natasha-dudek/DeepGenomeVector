from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import random
from copy import deepcopy
import numpy as np
import torch
import random

def hamming(target, b_pred):
	hs = []
	for i in range(len(b_pred)):
		hs.append(hamming_loss(target[i], b_pred[i]))
	return hs
	
def dom_confusion_matrix(b_pred, target, num_to_genome, genome_to_tax, test_tax_dict, genome_idx_test):
	
	"""
	pred = (baseline: corrupt_test_data) or (model: b_pred)
	target = uncorrupt_test_data
	"""
	
	num_0s = sum(sum(target == 0))
	num_1s = sum(sum(target == 1))
	p_0s = round(num_0s / (num_0s + num_1s)* 100, 1)
	p_1s = round(num_1s / (num_0s + num_1s)* 100, 1)
	
	domain_confusion = defaultdict(lambda: defaultdict(list))
	
	for i in range(len(b_pred)):
	
	    # Get taxonomic lineage for each genome
	
	    tax = genome_to_tax[test_tax_dict[genome_idx_test[i]]]
	    
	    phylum_mode = False
	    if phylum_mode:
	        taxonomy = tax.split(";")[1].split("p__")[1].strip("*")
	        if taxonomy == "Proteobacteria":
	            taxonomy = tax.split(";")[2].split("c__")[1].strip("*")
	    else:
	        taxonomy = tax.split(";")[0].split("k__")[1].strip("*")
	        
	    # CHANGED FROM PRED TO B_PRED
	    tn, fp, fn, tp = confusion_matrix(b_pred[i], target[i]).ravel()
	    acc = accuracy_score(target[i], b_pred[i])
	    
	    domain_confusion[taxonomy]['tn'].append(tn)
	    domain_confusion[taxonomy]['fp'].append(fp)
	    domain_confusion[taxonomy]['fn'].append(fn)
	    domain_confusion[taxonomy]['tp'].append(tp)
	    domain_confusion[taxonomy]['acc'].append(acc)
	
	avg_dict = defaultdict(list)
	for taxonomy in domain_confusion:
	    tn = sum(domain_confusion[taxonomy]['tn'])
	    fp = sum(domain_confusion[taxonomy]['fp'])
	    fn = sum(domain_confusion[taxonomy]['fn'])
	    tp = sum(domain_confusion[taxonomy]['tp'])
	    acc = (tn + tp) / (tn + fp + tp + fn)
	    out = tn, fp, fn, tp
	    out = [str(round(i/sum(out)*100, 1))+"%" for i in out]
	    acc = (tn + tp) / (tn + fp + tp + fn)
	    out.append(str(round(acc*100, 1))+"%")
	    avg_dict[taxonomy] = out
	
	truth = [str(p_0s)+"%", str(0)+"%", str(0)+"%", str(p_1s)+"%", '100.0%']
	avg_dict['Ground truth']=truth	
	
	confusion_df = pd.DataFrame.from_dict(avg_dict, orient='index', columns=['TN', 'FP', 'FN', 'TP', 'ACC'])
							
	return confusion_df

def generate_baseline(num_features, train_data, corrupt_test_data, mode, cluster_names):
	
	
	if mode != "base_random" and mode != "smart_random":
		raise ValueError ('mode must be set to either "base_random" or "smart_random"')
	
	# Gene counts of genomes in full training dataset
	gene_counts = np.sum(train_data.detach().numpy(), axis=1)
	b_pred = np.zeros(corrupt_test_data.shape)

	train_data2 = torch.tensor([i.numpy() for i in train_data]).float()
	corrupt_train = train_data2[:,:len(cluster_names)]
	
	tempy = deepcopy(corrupt_test_data)
	
	for i in range(len(corrupt_test_data)):

		#genome = deepcopy(corrupt_test_data[0])
		genome = tempy[i]

		if mode == "base_random":
			gene_count = sum(genome.numpy() == 1) 
			turn_on = int(gene_count*1.2 - gene_count)
			p_on = turn_on / num_features
			switch_idx = np.random.binomial(n=1, size=len(genome), p=p_on).astype(np.bool)
			
		elif mode == "smart_random":
		
			ko_prob = np.sum(corrupt_train.detach().numpy(), axis=0) / corrupt_train.shape[0]
			switch_idx = np.random.binomial(n=1, size=len(genome), p=ko_prob).astype(np.bool)
		
		genome[switch_idx] = 1
		b_pred[i] = genome
		

	return b_pred

def majority_baseline(corrupt_test_data):
	
	b_pred = np.zeros(corrupt_test_data.shape)
	
	for i in range(len(corrupt_test_data)):
		genome = deepcopy(corrupt_test_data[0])



def eval_binarize(pred, replacement_threshold):
	from sklearn.preprocessing import Binarizer
	return torch.LongTensor(Binarizer(threshold=replacement_threshold).fit_transform(pred))

def baseline1(corrupted_train, org_to_mod_to_kos, org_to_kos, tla_to_tnum, c_train_genomes, corrupted_test):
	
	# Use training set to get stats about gene count dist.
	def gene_dist(org_to_mod_to_kos, org_to_kos, tla_to_tnum, c_train_genomes):
	    gene_counts = []
	    for org in org_to_mod_to_kos:
	        tnum = tla_to_tnum[org]
	        if org in c_train_genomes and len(org_to_mod_to_kos[org]) >= 10 and len(org_to_kos[tnum]) >= 400: # make sure org is in training set, not test
	            if len(org_to_kos[tnum]) < 10:
	                print()
	                print (tnum, len(org_to_kos[tnum]))
	                print(len(org_to_mod_to_kos[org]))
	            gene_counts.append(len(org_to_kos[tnum]))
	    return gene_counts
	gene_counts = gene_dist(org_to_mod_to_kos, org_to_kos, tla_to_tnum, c_train_genomes)

	n_features = int(corrupted_test.shape[1]/2)
	baseline1 = torch.zeros_like(corrupted_test[:,:n_features])
		
	for i in range(baseline1.shape[0]):
	    # get idx of on bits in corrupted vector 
	    on_pos = [int(s) for s in (corrupted_test[i,:n_features] == 1).nonzero()]
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

def glom_confusion(uncorrupted, baseline1):
	tns = []
	fps = []
	fns = []
	tps = []
	
	for idx, row in enumerate(uncorrupted):
	    tn, fp, fn, tp = confusion_matrix(row, baseline1[idx]).ravel()
	    tns.append(tn)
	    fps.append(fp)
	    fns.append(fn)
	    tps.append(tp)
	    
	total = sum(tns+fps+fns+tps)

	return tns, fps, fns, tps, total

def baseline2(corrupted_train, org_to_mod_to_kos, org_to_kos, tla_to_tnum, c_train_genomes, corrupted_test):
	n_features = int(corrupted_train.shape[1]/2)
	# Use training set to calculate stats about prob bits being on
	uncorrupted = corrupted_train[:,n_features:] # uncorrupted
	per_colum = torch.sum(uncorrupted, dim=0) # sum of each column
	highest_prob = list(torch.argsort(per_colum, descending=True).numpy())
	
	def gene_dist(org_to_mod_to_kos, org_to_kos, tla_to_tnum):
	    gene_counts = []
	    for org in org_to_mod_to_kos:
	        tnum = tla_to_tnum[org]
	        if org in c_train_genomes and len(org_to_mod_to_kos[org]) >= 10 and len(org_to_kos[tnum]) >= 400:
	            if len(org_to_kos[tnum]) < 10:
	                print()
	                print (tnum, len(org_to_kos[tnum]))
	                print(len(org_to_mod_to_kos[org]))
	            gene_counts.append(len(org_to_kos[tnum]))
	    return gene_counts
	
	gene_counts = gene_dist(org_to_mod_to_kos, org_to_kos, tla_to_tnum)
	
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


def baseline4(corrupted_train, corrupted_test, tla_to_tnum, org_to_kos, c_train_genomes):
	n_features = int(corrupted_train.shape[1]/2)
	
	### Find smallest genome in train set
	unique_train = list(set(c_train_genomes))
	tla_size = []
	for tla in unique_train:
	    tnum = tla_to_tnum[tla]
	    tla_size.append([tla, len(org_to_kos[tnum])])
	sorted_tla_size = sorted(tla_size, key=lambda x: x[1], reverse=False)
	smallest_tla = sorted_tla_size[0][0] # tla = 'hed'
	print("smallest_tla",smallest_tla)
	# row index of smallest genome in train set
	start = c_train_genomes.index(smallest_tla) # hed = Hoaglandella endobia, Gammaproteobacteria
	smallest_uncorrupted = corrupted_train[start,n_features:]

	# Create baseline for test set
	baseline4 = torch.Tensor(np.tile(smallest_uncorrupted, (corrupted_test.shape[0], 1)))
	
	return baseline4.long()

def baseline5(corrupted_train, corrupted_test, tla_to_tnum, org_to_kos, c_train_genomes):
	n_features = int(corrupted_train.shape[1]/2)
	
	### Find smallest genome in train set
	unique_train = list(set(c_train_genomes))
	tla_size = []
	for tla in unique_train:
	    tnum = tla_to_tnum[tla]
	    tla_size.append([tla, len(org_to_kos[tnum])])
	sorted_tla_size = sorted(tla_size, key=lambda x: x[1], reverse=True)
	largest_tla = sorted_tla_size[0][0] # tla = hed
	print("largest_tla",largest_tla)	# row index of smallest genome in train set
	start = c_train_genomes.index(largest_tla) # hed = Hoaglandella endobia, Gammaproteobacteria
	largest_uncorrupted = corrupted_train[start,n_features:]

	# Create baseline for test set
	baseline5 = torch.Tensor(np.tile(largest_uncorrupted, (corrupted_test.shape[0], 1)))
	
	return baseline5.long(), largest_tla





