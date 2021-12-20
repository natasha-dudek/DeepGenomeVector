from collections import defaultdict
import random
import sys

import numpy as np
import torch

def heart_of_corruption(org_to_mod_to_kos, org, n_kos_tot, all_kos, n_mods):
	"""
	For each genome vector, keep the KO's in n_mods modules. All other bits will be converted to zeros.
	
	Arguments:
	org (str) -- tla for genome (e.g.: "aha")
	n_max (int) -- the maximum number of mods to select for corrupted version of any given genome
	
	Returns:
	corrupted (np array) -- vector representing corrupted version of a genome
	"""
	keeps = random.sample(list(org_to_mod_to_kos[org].keys()), n_mods)			
	idxs = []
	for mod in keeps:
		for ko in org_to_mod_to_kos[org][mod]:
			idxs.append(all_kos.index(ko))

	# create corrupted version of genome that only has those mods
	corrupted = np.zeros(n_kos_tot)
	for i in idxs:
		corrupted[i] = 1

	return corrupted, keeps

def corrupt(train_data, train_genomes, n_corrupt, tnum_to_tla, org_to_mod_to_kos, all_kos, mod_to_ko_clean, n_mods):
	"""
	Perform corruptions on a set of genome vector. This means keep the KO's in n_mods modules for each genome vector. All other bits will be converted to zeros.
	
	Note: creates corrupted + matching uncorrupted tensor of genomes, in that order
	Note: only genomes with >= 1 module are included in the output
	Note: uses "cleaned" modules from mod_to_ko_clean  
		I.e. most common set of KOs per module, rather than 20 variants of each mod
	
	Arguments:
	train_data (tensor) -- rows = uncorrupted genomes, columns = KOs
	train_genomes (list) -- names of genomes in train_data (e.g.: "T03060")
	n_corrupt (int) -- number of corrupted versions to make of each genome
	tnum_to_tla (dict) -- maps tnum (e.g.: "T03060") to tla (e.g.: "Red")
	
	Returns:
	output (tensor) -- corrupted + uncorrupted genomes (each genome's two versions are concatenated in a row)
	c_train_genomes (list) -- names of genomes in the order they appear in output
	"""
		
	output = [] 
	c_train_genomes = []
	n_kos_tot = train_data.shape[1]
	input_mods = []
	
	line_counter = 0
	for i, tnum in enumerate(train_genomes):
		org = tnum_to_tla[tnum]
		n_tot_mods = len(org_to_mod_to_kos[org]) # number of modules in the genome 
		
		n_corrupted = 0
		uncorrupted = train_data[i]
		while n_corrupted < n_corrupt: 
			c_train_genomes.append(org)
			
			corrupted, in_mods = heart_of_corruption(org_to_mod_to_kos, org, n_kos_tot, all_kos, n_mods)

			genome_out = np.concatenate((corrupted, uncorrupted), axis=None)
			output.append(genome_out)
			input_mods.append(in_mods)
			line_counter += 1
			n_corrupted += 1
			
	return torch.Tensor(output), c_train_genomes, input_mods

def new_corrupt(BASE_DIR, train_data, train_genomes, n_corrupt, tnum_to_tla, tla_to_mod_to_kos, all_kos, mod_to_ko_clean, n_mods):
	"""
	Generate new genome corruptions from scratch
	
	Arguments:
		BASE_DIR (str) -- path to working directory
		train_data (numpy.ndarray) -- training data. Rows are genomes, columns are genes/KOs. 1's denote presence of a gene in the genome, 0's denote absence
		train_genomes (list) -- tnums of genomes in the training set
		n_corrupt (int) -- number of corrupted version to make of each genome
		tnum_to_tla (dict) -- for each genome, converts tnum to tla
		tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
		all_kos (list) -- list of all KOs in the dataset
		mod_to_ko_clean (dict )-- the functions of many modules can be "completed" by different sets of genes. Here we choose to represent each module by the most common set of genes. Dict maps each module (e.g.: 'K00001') to a list of genes (e.g.: ['K00845', ..., 'K00873'])
		n_mods (int) -- maximum number of mods to keep per corrupted genome
	
	Returns:
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1 denotes gene encoded by genome, 0 denotes absence.
		c_train_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
		train_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_train_genomes)
		corrupted_test (tensor) -- corrupted test data. Rows are genomes, columns are genes. 1 denotes gene encoded by genome, 0 denotes absence.
		c_test_genomes (list) -- tnum corresponding to each row (genome) of corrupted_test
		test_input_mods (list) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_test_genomes)
	"""
	print ("Creating new corruptions")
	print (str(n_corrupt),"corruptions per genome")
	print ("With",str(n_mods),"modules as input per genome")
	print ("Saved as",date_to_save)

	# Do corruptions from training and then test sets
	corrupted_train, c_train_genomes, train_input_mods = corrupt.corrupt(train_data, train_genomes, n_corrupt, tnum_to_tla, tla_to_mod_to_kos, all_kos, mod_to_ko_clean, n_mods)
	corrupted_test, c_test_genomes, test_input_mods = corrupt.corrupt(test_data, test_genomes, n_corrupt, tnum_to_tla, tla_to_mod_to_kos, all_kos, mod_to_ko_clean, n_mods)
	
	# Save!!!
	torch.save(corrupted_train, BASE_DIR+"corrupted_train_"+date_to_save+".pt")
	torch.save(c_train_genomes, BASE_DIR+"c_train_genomes_"+date_to_save+".pt")
	torch.save(corrupted_test, BASE_DIR+"corrupted_test_"+date_to_save+".pt")
	torch.save(c_test_genomes, BASE_DIR+"c_test_genomes_"+date_to_save+".pt")
	torch.save(train_input_mods, BASE_DIR+"train_input_mods_"+date_to_save+".pt")
	torch.save(test_input_mods, BASE_DIR+"test_input_mods_"+date_to_save+".pt")
	
	return corrupted_train, c_train_genomes, train_input_mods, corrupted_test, c_test_genomes, test_input_mods
	
def load_corrupt(BASE_DIR, date_to_load):
	"""
	Loads corrupted genome vectors from file
	
	Arguments:
		BASE_DIR (str) -- path to working directory
		date_to_load (str) -- name of file to load
		
	Returns:
		corrupted_train (tensor) -- corrupted training data. Rows are genomes, columns are genes. 1 denotes gene encoded by genome, 0 denotes absence.
		c_train_genomes (list) -- tnum corresponding to each row (genome) of corrupted_train
		train_input_mods (list of lists) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_train_genomes)
		corrupted_test (tensor) -- corrupted test data. Rows are genomes, columns are genes. 1 denotes gene encoded by genome, 0 denotes absence.
		c_test_genomes (list) -- tnum corresponding to each row (genome) of corrupted_test
		test_input_mods (list) -- lists of the mods that were retained during the corruption process (in same order as genome rows / c_test_genomes)
	"""
	print("Loading corrupted genome vectors from "+date_to_load)
	corrupted_train = torch.load(BASE_DIR+"corrupted_train_"+date_to_load+".pt")
	c_train_genomes = torch.load(BASE_DIR+"c_train_genomes_"+date_to_load+".pt")
	corrupted_test = torch.load(BASE_DIR+"corrupted_test_"+date_to_load+".pt")
	c_test_genomes = torch.load(BASE_DIR+"c_test_genomes_"+date_to_load+".pt")
	train_input_mods = torch.load(BASE_DIR+"train_input_mods_"+date_to_load+".pt")
	test_input_mods = torch.load(BASE_DIR+"test_input_mods_"+date_to_load+".pt")
	
	return corrupted_train, c_train_genomes, train_input_mods, corrupted_test, c_test_genomes, test_input_mods
	
