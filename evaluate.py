from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import random
from copy import deepcopy
import numpy as np
import torch


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











