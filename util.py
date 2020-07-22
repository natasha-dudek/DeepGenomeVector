import pandas as pd
import torch 
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from genome_embeddings import data_viz
import os, sys
from collections import defaultdict
import random
import pickle

def genome_to_tax(df):
	file = open("/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/selected_kegg.txt", "r").readlines()
	file = map(str.strip, file)
		
	genome_to_tax = {}
	line_counter = 0
	for s in file:
		if line_counter < 4:
			line_counter += 1
			continue
		genome_to_tax[s.split()[2]] = s.split(" ")[3]
	
	#tax = [genome_to_tax[i] for i in list(df.index)] # list(df.index) is genome names T01278

	return genome_to_tax
	
def load_data(path, data_source):
	"""
	Load data from either KEGG or get_homologues datasets
	
	Expecting a matrix where: 
		header = gene names
		first column = genome names
	"""
	
	if data_source == "get_homologues":
		# load pangenome matrix as pandas df
		df = pd.read_csv(path+"pangenome_matrix_t0.tab", sep='\t', index_col=0)
	elif data_source == "kegg":
		df = pd.read_csv(path+"genome_to_mod.csv", index_col=0)
	else:
		raise ValueError('Unsure what kind of data you are trying to load. Accepted types are "kegg" and "get_homologues"')
	
	# Any value greater than 1 should be set to 1 --- presence/absence not abundance
	df[df > 1] = 1 
	
	# Drop KOs that occur fewer than ngleton times from dataset
	#df = df[df.columns[df.sum()>= ngeltons]]

	# Get column names and make some handy plots to better understand data
	if data_source == "get_homologues":
		cluster_names = df.columns.values.tolist()
		cluster_names = [i.split(".faa")[0] for i in cluster_names][1:]
	elif data_source == "kegg":
		cluster_names = df.columns.values.tolist()
		#data_viz.taxa_per_cluster(df)
		#data_viz.clusters_per_taxa(df)
		
	return df, cluster_names


def corrupt(data, num_corruptions, corruption_fraction, cluster_names, mode, path):
	#### USE THIS ONE IF DATA IS TENSOR (not tensor subset)
	
	"""
	Stochastically drop KO's / modules from genomes
	
	Arguments:
	data (df) -- train or test dataset
	num_corruptions (int) -- number of corrupted outputs to produce per input genome
	corruption_fraction (float) -- what % of KO's/mods to drop
	
	Returns:
	out (tensor) -- training set with corrupted genomes and then uncorrupted genomes in each row
	genome_idx (dict) -- maps genome idx in corrupt_train / corrupt_test to genome ID
		E.g.: genome_idx[i] -> 'T012839'
	"""

	if mode != "train" and mode != "test":
		raise ValueError ("mode must either be 'train' or 'test'")

	data = torch.tensor(data.values) 

	num_genomes = data.shape[0] # number of genomes in the train ds
	out = np.zeros(shape=(num_genomes*num_corruptions,data.shape[1]*2))
	
	# Create dict that can trace each genome back to original genome index 
	genome_idx = {}
	genome_counter = 0
	
	# Iterate through original genomes ---> produce corrupt versions 
	for s in range(num_genomes):
		# get indices of KO's present 
		ko_idx = np.argwhere(data[s] == 1).tolist()[0]
		uncorr_idx = [(i + data.shape[1]) for i in ko_idx]
		# generate num_corruptions corrupted genomes from original genome
		for i in range(num_corruptions):
			# random sampling of gene idxs without replacement
			keeper_idx = random.sample(ko_idx, int(len(ko_idx)*corruption_fraction))
			# retain only keeper genes
			out[genome_counter][keeper_idx] = 1
			# Then add uncorrupted genome
			out[genome_counter][uncorr_idx] = 1
			genome_idx[genome_counter] = s
			genome_counter += 1
	
	out = torch.FloatTensor(out)
	
	if mode == "train":
		torch.save(out, path+"corrupted_train_07-17-20.pt")
		np.savetxt(path+"corrupted_train_07-17-20.txt", out.numpy())
		torch.save(genome_idx, path+"genome_idx_train_07-17-20.pt")
	else:
		torch.save(out, path+"corrupted_test_07-17-20.pt")
		np.savetxt(path+"corrupted_test_07-17-20.txt", out.numpy())
		torch.save(genome_idx, path+"genome_idx_test_07-17-20.pt")
			 
	return out, genome_idx


	
def train_test_split(df, cluster_names, n_test, path, genome_to_num):#, batch_size, num_corruptions, corruption_fraction):
	"""
	Loads data, splits data into train and test sets, returns a few basic stats
	
	Arguments:
	path -- path to file containing data to load
	ngletons -- drop any inputs occur fewer than "ngletons" times across all genomes 
	n_test -- number of genomes to go into the test set
	batch_size -- batch size for training and test set dataloaders
	data_source -- what time of data you're trying to load ['get_homologues' | 'kegg']
	
	Returns:
	train -- full training set (torch tensor)
	test -- full test set (torch tensor)
	num_clusters -- the number of features in your dataset 
	num_genomes -- the number of genomes in your dataset
	cluster_names -- the names of features in your dataset
	"""
		
	# Get rid of singletons or n-gletons in whole set
	# Note: if you do this after train-test split the training and test set will
	# have diff numbers of features which will break the model 
	tensor_df = torch.tensor(df.values) # convert df to tensor	
#	mask = tensor_df.sum(axis=0) < ngeltons
#	column_indices = np.where(mask)[0]
#	tensor_df = tensor_df[:,~mask]
	
	# Split train and test set
	test_size = int(round(n_test*tensor_df.shape[0]))
	train_size = int(tensor_df.shape[0] - test_size)
	print(train_size, test_size, train_size+test_size)
	train, test = random_split(tensor_df, [train_size, test_size])
	
	torch.save(train, path+"uncorrupted_train.pt")
	torch.save(test, path+"uncorrupted_test.pt")
	
	return train, test

def dataloaders(train_data, test_data, batch_size, test_size, num_features):

	"""
	Creates dataloaders for corrupted and uncorrupted genomes
	
	Arguments:
	train_data -- df of training data 
		Each row represents a corrupted version of a genome + the uncorrupted genome concatenated together
	test_data -- df of test data
		Each row represents a corrupted version of a genome + the uncorrupted genome concatenated together
	batch_size -- batch size for training set
	test_size -- batch size for test set
	num_features -- number of features / genes
	
	Returns:
	loaders -- dict of dataloaders 
		keys = "train" | "test"
		values = TensorDatasets (x_train - corrupted, y_train - uncorrupted)
	"""
	# Tutorial: https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
	
	# tensors with corrupted genomes and then uncorrupted genomes in each row, torch.Tensor
	x_train = train_data[:,:num_features] # corrupted genomes in first half of matrix
	y_train = train_data[:,num_features:] # uncorrupted in second half
	x_test = test_data[:,:num_features]
	y_test = test_data[:,num_features:]
	
	train_ds = TensorDataset(x_train, y_train)
	train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=True)
	
	test_ds = TensorDataset(x_test, y_test)
	test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
	
	loaders={}
	loaders['train'] = train_dl
	loaders['test'] = test_dl
	
	return loaders

def balanced_split(df, n_test, genome_to_tax, num_to_genome, path=None, genome_idx_train=None):
	"""
	Creates phylogenetically balanced train-test split at domain + phylum levels
	
	For taxonomic groups with >=10 genomes, split those 10 by n_test
	For groups with <10 and > 1 genomes, split them 50-50 into train/test
	For groups with only 1 genome, put it in the train set
	Note: the fraction of train vs test data will not be exactly equal to n_test 
	
	Arguments:
	df -- df produced by util.load_data function
	n_test -- Fraction of data to go into test set, number between 0 and 1 
	genome_to_tax -- dict mapping each genome to taxonomic lineage
	num_to_genome -- dict mapping a unique number to each genome
	path -- where to save train-test split generated by the function
	
	Returns / saves:
	train_split --  df containing training data (rows = genomes)
	test_split --- df containing test data (rows = genomes)
	"""
		
	# Dict where keys are domains and values are a list of occurences of each phylum within the domain
	# E.g.: within Bacteria [Gammaprot, Bacteroidetes, Gammaprot, Actino, ...]
	# get tax of genomes in ds (genome_to_tax has 3474 genomes, we're using only 3432 in ds)
	
	genome_to_tax2 = {}
		
	if genome_idx_train is not None:
		for i in df.index:
			genome_num = genome_idx_train[i]
			genome_id = num_to_genome[genome_num] 
			tax = genome_to_tax[genome_id]
			genome_to_tax2[i] = tax	
	else:
		for i in df.index:
			genome_to_tax2[i] = genome_to_tax[i]
	
	tensor_df = torch.tensor(df.values)
	
	# Default dict where keys are phyla and values are a list of genomes per phylum
	phylum_to_genomes = defaultdict(list)
	for genome in genome_to_tax2:
		phylum = genome_to_tax2[genome].split(";")[1][3:].strip("*")
		if phylum == "Proteobacteria":
			phylum = genome_to_tax2[genome].split(";")[2][3:].strip("*")
		phylum_to_genomes[phylum].append(genome)	
	
	#Default dict where keys are domains and valyes are a list of occurences of each phylum within the domain
	# E.g.: within Bacteria [Gammaprot, Bacteroidetes, Gammaprot, Actino, ...]
	# get tax of genomes in ds (genome_to_tax has 3474 genomes, we're using only 3432 in ds)
	bacteria = defaultdict(lambda: defaultdict(int))
	for i in genome_to_tax2:
		domain = genome_to_tax2[i].split(";")[0][3:] # [3:] gets rid of k__ in k__Phylum_name
		if domain == "TM6":
			domain = "Bacteria"
		phylum = genome_to_tax2[i].split(";")[1][3:].strip("*")
		if phylum == "Proteobacteria":
			phylum = genome_to_tax2[i].split(";")[2][3:].strip("*")
		bacteria[domain][phylum] += 1


	# select genome IDs (e.g.: T121587) for train / test set
	test_train = {'train': [], 'test': []}
	for domain in bacteria:
		num_test = round(sum(bacteria[domain].values())*n_test)
		num_train = sum(bacteria[domain].values()) - num_test
		
		for phylum in bacteria[domain]:
			
			num_genomes = bacteria[domain][phylum]
			
			#print(phylum_to_genomes[phylum])
			
			if 1 < num_genomes < 10:
				# 50-50 split to training vs test set
				p_test = round(bacteria[domain][phylum]*0.5)
				p_train = bacteria[domain][phylum] - p_test
				
			elif num_genomes == 1:
				#p_test = round(bacteria[domain][phylum]*1)
				#p_train = bacteria[domain][phylum] - p_test
				p_test = 0
				p_train = 1
				
			else:
				# phylum: [T1230, T327891, T32780]
				p_test = round(bacteria[domain][phylum]*n_test)
				p_train = bacteria[domain][phylum] - p_test
			
			#print(domain, phylum, num_genomes, p_train, p_test)
			p_train_ds = random.sample(phylum_to_genomes[phylum], p_train)
	
			p_test_ds = []
			for genome in phylum_to_genomes[phylum]:
				if genome not in p_train_ds:
					p_test_ds.append(genome)
					
			#print (len(p_train_ds), len(p_test_ds))
			test_train['train'].extend(p_train_ds)
			test_train['test'].extend(p_test_ds)

	train_split = {}
	test_split = {}
	for i in range(len(tensor_df)):
			
		if genome_idx_train is None:
			#print("why am i here?")
			sys.stdout.flush()
			sys.stderr.flush()
			genome = num_to_genome[i] 
		else:
			#print("this is where I should be")
			sys.stdout.flush()
			sys.stderr.flush()
			genome_num = genome_idx_train[i]
			genome = num_to_genome[genome_num] 
			
		if genome in test_train['train']:
			train_split[genome] = tensor_df[i].tolist()
		else:
			test_split[genome] = tensor_df[i].tolist()
	
	train_df = pd.DataFrame.from_dict(train_split, orient='index')
	test_df = pd.DataFrame.from_dict(test_split, orient='index')
	
	# save files if not doing hp tuning
	if path is not None:
		train_df.to_csv(path+"uncorrupted_train_balanced.csv", index_label=0)
		test_df.to_csv(path+"uncorrupted_test_balanced.csv", index_label=0)
	
#	# some helpful stats
#	num_train = len(test_train['train'])
#	num_test = len(test_train['test'])
#	p_train = round(num_train/ (num_train+num_test) * 100)
#	p_test = 100 - p_train
#	print(("Number of training examples: %s (%s%%), Number of test examples: %s (%s%%)") % (num_train, p_train, num_test, p_test))
#	
	return train_df, test_df

def bacteria_only(data, train_genomes, genome_to_tax):
	"""
	data - df
	genome_idx_dict - genome_idx_train or genome_idx_test
   	
   	
   	out - numpy array
	"""
	
	tax_dict = {}
	
	out = np.zeros_like(data)
	
	bact_counter = 0
	for s in range(len(data)):
		#orig_idx = genome_idx_train[s] # corrupt 9 to uncorrupt 0
		
		t_num = train_genomes[s]
		tax = genome_to_tax[t_num] # list  of dataframe index of row orig_idx
		
		domain = tax.split(";")[0]
		if domain != "k__Bacteria": continue
		
		out[bact_counter] = data.iloc[s]
		
		tax_dict[bact_counter] = t_num
		
		bact_counter += 1
	
	# delete out rows that weren't filled
	out = out[:bact_counter,:]
	
	#out = torch.tensor(out).float()   
	
	return pd.DataFrame(out), tax_dict

def cirriculum_load(train_data, test_data, batch_size, test_size, cluster_names):
	"""
	Arguments:
		train_data -- tensor of training data 
			Each row represents a corrupted version of a genome + the uncorrupted genome concatenated together
		test_data -- tensor of test data
			Each row represents a corrupted version of a genome + the uncorrupted genome concatenated together
		batch_size -- batch size for training set
		test_size -- batch size for test set
		cluster_names -- list of names of clusters, used to know how many clusters there are / where to split corrupted vs uncorrupted
	
	Returns:
	loaders -- dict of dataloaders 
		keys = "train" | "test"
		loaders['train'][0] = small genomes
		loaders['train'][1] = medium genomes
		loaders['train'][2] = large genomes
		loaders / values = TensorDatasets (x_train - corrupted, y_train - uncorrupted)
	"""
	
	# First split training data into three groups based on gene count
	# Small, medium, and large gene counts
	
	gene_count = list(train_data.sum(axis=1).numpy())
	
	# smallest to largest gene count, tuple of (gene_count, index)
	gene_count = sorted(((count, idx) for idx, count in enumerate(gene_count)), reverse=False)
	
	len_split = int(len(gene_count)/3)
	len_split
	small_idx = [i[1] for i in gene_count[:len_split]]
	medium_idx = [i[1] for i in gene_count[len_split:len_split*2]]
	large_idx = [i[1] for i in gene_count[len_split*2:]]
	
	small_train = train_data[small_idx]
	medium_train = train_data[medium_idx]
	large_train = train_data[large_idx]

	x_small_train = small_train[:,:len(cluster_names)] # corrupted genomes in first half of matrix
	y_small_train = small_train[:,len(cluster_names):] # uncorrupted in second half
	
	x_medium_train = medium_train[:,:len(cluster_names)] # corrupted genomes in first half of matrix
	y_medium_train = medium_train[:,len(cluster_names):] # uncorrupted in second half
	
	x_large_train = large_train[:,:len(cluster_names)] # corrupted genomes in first half of matrix
	y_large_train = large_train[:,len(cluster_names):] # uncorrupted in second half
	
	x_test = test_data[:,:len(cluster_names)]
	y_test = test_data[:,len(cluster_names):]
	
	small_train_ds = TensorDataset(x_small_train, y_small_train)
	small_train_dl = DataLoader(small_train_ds, batch_size=batch_size, drop_last=False, shuffle=True)

	medium_train_ds = TensorDataset(x_medium_train, y_medium_train)
	medium_train_dl = DataLoader(medium_train_ds, batch_size=batch_size, drop_last=False, shuffle=True)

	large_train_ds = TensorDataset(x_large_train, y_large_train)
	large_train_dl = DataLoader(large_train_ds, batch_size=batch_size, drop_last=False, shuffle=True)

	test_ds = TensorDataset(x_test, y_test)
	test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
	
	loaders={}
	loaders['train'] = [small_train_dl, medium_train_dl, large_train_dl]
	loaders['test'] = test_dl
	
	return loaders

def remove_rare(train_orig, test_orig, cluster_names, thresh):
	"""
	Remove rare features from dataset (train + test)
	
	Arguments:
	train_orig (dataframe) -- train dataset before producing corrupted genomes 
	test_orig (dataframe) -- test dataset before producing corrupted genomes
	thresh (int) -- genes occur fewer than this many times in train dataset 
		will be removed from train and test datasets 
	
	Returns:
	train_orig2 (dataframe) -- train_orig minus rare features
	test_orig2 (dataframe) -- test_orig minus rare features 
	"""
	
	mask = [i for i,val in enumerate(train_orig.sum()) if val < thresh ] # indices of features to drop
	cat_df = pd.concat([train_orig, test_orig])
	cat_df = cat_df.drop(cat_df.columns[mask], axis=1)
	train_orig = cat_df.head(len(train_orig))
	test_orig = cat_df.tail(len(test_orig))
	
	cn2 = [val for i,val in enumerate(cluster_names) if i not in mask]
	
	return train_orig, test_orig, cn2

def date_time():
	from datetime import datetime
	
	dateTimeObj = datetime.now()
	date = str(dateTimeObj.day)+"-"+str(dateTimeObj.month)+"-"+str(dateTimeObj.year)
	time = "_"+str(dateTimeObj.hour)+"-"+str(dateTimeObj.minute)+"-"+str(dateTimeObj.second)
	return date+time

def log_results(roc, optim_lc, perf_lc, flags, model, cm):

	import git
	
	datetime = date_time()
	save_path = "log/"+datetime+"/"
	
	os.system("mkdir "+save_path)
	
	settings = open(save_path+"settings.txt","w")
	
	# Save hash for git repo
	repo = git.Repo()
	repo_hash = repo.head.object.hexsha
	settings.write("Repo hash: "+repo_hash+"\n")

	# Save flags from Namespace (hyperparams, etc)
	for i in vars(flags):
		settings.write(i+": "+str(vars(flags)[i])+"\n")
	
	settings.write("\nconfusion matrix: \n")
	cm.to_string(settings)
	settings.close()
	
	# Save useful figures
	roc.savefig(save_path+"roc.png", dpi=200)
	optim_lc.savefig(save_path+"optim_lc.png", dpi=200)
	perf_lc.savefig(save_path+"perf_lc.png", dpi=200)
	
	# Save model
	torch.save(model, save_path+"model.pt")
	
	
	
	
	


