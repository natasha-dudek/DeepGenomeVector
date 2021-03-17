import numpy as np
import pickle
import re
from collections import defaultdict
import torch

def genomes2include(mode):
	"""
	Figure out which genomes I want to include in my dataset based on those in selected_kegg.txt
	
	Arguments: None
	
	Returns: 
	tla_to_tnum (dict) -- converts three-letter abbreviations (e.g.: eco) to t-numbers (e.g.: T04989) 
	keepers (list) -- list of t-numbers to keep (e.g.: T04989)
	"""
	
	if mode == 'CC':
		path = './'
	else:
		path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/'
	
	# Use phylogenetically thinned list in selected_genomes.txt AND filter out non-bacteria
	path = path+"selected_kegg.txt"
	file = open(path).readlines()
	file = list(map(str.strip, file))
	
	# The keepers list ends up having a few genomes that do not get used
	# This is bc they have no full modules
	keepers = [] 
	# Create dict that converts tla (e.g.: Pea) -> t_num (e.g.: T321890)
	tla_to_tnum = {}
	
	line_counter = 0
	for s in file:
		if line_counter < 4:
			line_counter += 1
			continue
		tla = s.split()[1]
		t_num = s.split()[2]
		tax = s.split()[3]
	   
		if "k__Bacteria" in tax:
			keepers.append(t_num)
			tla_to_tnum[tla] = t_num
	
	print("Total number of bacterial genomes in dataset: {}".format(len(keepers)))
	
	return tla_to_tnum, keepers

def load_kos(tla_to_tnum, mode):
	"""
	Load mapping of genome to KOs encoded
	
	Arguments:
	tla_to_tnum (dict) -- converts three-letter abbreviations (e.g.: eco) to t-numbers (e.g.: T04989) 
	
	Returns:
	org_to_kos (dict of lists) -- for each organism, a list of KOs encoded by the genome 
	n_kos_tot (int) -- number of KOs in org_to_kos
	"""

	### load each genome as a list of KOs
	if mode == 'CC':
		path = './annotations/'
	else:
		path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/annotations/'
	
	
	#path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/annotations/annotations_list.txt"
	master_file = open(path+"annotations_list.txt").readlines()
	master_file = list(map(str.strip, master_file))
	
	### Create dict mapping genomes to encoded KOs
	# key = t_num 
	# ko = list of all KOs annotated in genome
	org_to_kos = {}
	for i in master_file:
		file = open(path+i).readlines()
		file = list(map(str.strip, file))
	   
		org = i.split("_")[0]
	   
		try:
			t_num = tla_to_tnum[org]
		except KeyError: continue # phylogenetically thinned set, not all will be present in dict (e.g. "Pea")
		#if t_num not in keepers: continue
	   
		kos = []
		for s in file:
			if "<a href=" in s:
				x = s.split()[2]
	#			 if "K00668" in s:
	#				 print("s",s)
	#				 print("x", x)
	#				 print()
				if re.match(r'[K]\d{5}', x): 
					kos.append(x) #[K]\d{5}
				   
		org_to_kos[t_num] = kos   

	# Create unique list of all KOs (all_kos) 
	all_kos = []
	for t_num in org_to_kos:
		all_kos.extend(org_to_kos[t_num])
	all_kos = list(set(all_kos))
	n_kos_tot = (len(all_kos))
	print("Total number of KOs in dataset: {}".format(n_kos_tot))

	return org_to_kos, n_kos_tot, all_kos

def load_mods(mode):
	"""
	Load mapping of genomes to modules and KOs encoded per module

	Arguments:
	
	Returns:
	org_to_mod_to_kos (dict of dict of list) -- three letter genome ID (e.g.: "has") to modules endoded to list of KOs per module
	mod_sets (defaultdict of defaultdict of int) -- for each module, all alternative pathways and their counts 
	"""
	# Load mapping from organism (tla, e.g.: Pea) to complete modules encoded to KOs in each module
	
	if mode == 'CC': path = './'
	else: path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/'
	with open(path+'org_to_mod_to_kos.pkl', 'rb') as f:
		org_to_mod_to_kos = pickle.load(f)
		
	# Create dict: mod_sets
	# For each module, have a counter (dict) of each unique string of KOs that can make up the module + # occurences 
	# mod_sets['M00001'] = {'K00001_K04398': 5, 'K00002_K23456': 10}
	mod_sets = defaultdict(lambda: defaultdict(int))
	for org in org_to_mod_to_kos:
		for mod in org_to_mod_to_kos[org]:
			ko_str = "_".join(org_to_mod_to_kos[org][mod])
			mod_sets[mod][ko_str] += 1
	
	# # organism "lkm" module "M00083" has a K0 "K00668" that is specific to fungi
	# It does not appear in bacterial genomes (verified for this set) 
	# Its presence in org_to_mod_to_kos will break code later on if not removed
	bad_ko = org_to_mod_to_kos["lkm"]["M00083"].index("K00668")
	del org_to_mod_to_kos["lkm"]["M00083"][bad_ko]
	
	return org_to_mod_to_kos, mod_sets



def make_tensor(org_to_mod_to_kos, org_to_kos, n_kos_tot, tla_to_tnum, all_kos, save=False):
	"""
	Convert org_to_kos dict to a tensor
	
	Arguments: 
	org_to_mod_to_kos ()
		
	Returns:
	data (tensor) -- rows are genomes, columns are KOs 
	genome_order (list) -- in the same order as data tensor, list of genomes IDs
	"""
	
	n_genomes = len(org_to_mod_to_kos)
	data = np.zeros(shape=(n_genomes,n_kos_tot))
	
	genome_order = []
	for i, org in enumerate(org_to_mod_to_kos):
		genome_order.append(org)
		for j in range(n_kos_tot):
			try:
				tla = tla_to_tnum[org]
				if all_kos[j] in org_to_kos[tla]:
					data[i,j] = 1
				else: pass
			except:
				print(i, org, tla, j)
				
	data = torch.tensor(data)
	
	return data, genome_order
	
def train_test_split(keepers):
	"""
	Duplicate train-test split from DAE
	
	Arguments:
		keepers (list) -- tnums of genomes to get
	
	Returns:
		train_genomes (list) -- list of genomes for training set
		test_genomes (list) -- list of genomes for test set
	"""

	### Create train-test split (ideally identical to former split)
	# Old original split contains euk and arch
	import pandas as pd
	path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_embeddings/data/uncorrupted_test_balanced.csv"
	test_orig = pd.read_csv(path, index_col=0) 
	test_genomes_orig = test_orig.index.to_list()
	
	train_genomes = []
	test_genomes = []
	for genome in keepers: 
		if genome not in test_genomes_orig:
			train_genomes.append(genome)
		elif genome in test_genomes_orig:
			test_genomes.append(genome)
			
	return train_genomes, test_genomes

def prep_data(list_genomes, all_kos, org_to_kos, mode):
	"""
	Creates a tensor for training / test data
	
	Arguments:
		list_genomes (list) -- t_nums to be included in the tensor (i.e. train_genomes or test_genomes)
		all_kos (list) -- all KOs that exist in the dataset
		org_to_kos (dict) -- keys = t_nums, values = list of KOs encoded by genome
		mode (str) -- used to save data to file ["test" | "train"] 
	
	Returns:
		data (np array) -- rows = genomes, columns = KOs, 1 = KO present in genome, 0 = KO absent in genome
	"""
	
	#assert (mode == "test" or mode == "train")
	
	data = np.zeros(shape=(len(list_genomes),len(all_kos)))
		
	for i, t_num in enumerate(list_genomes): # org is something like 'T03060'
		for j, ko in enumerate(all_kos):
			if ko in org_to_kos[t_num]:
				data[i,j] = 1
			else: pass

	return data


def clean_kos(mod_sets):
	# Select most "popular" version of each module, store it in dict: mod_to_ko_clean
	# mod_to_ko_clean['M00003'] = ['K00001', 'K00002', etc] <- most common variant of M00003
	mod_to_ko_clean = {}
	for mod in mod_sets:
		max_count = 0
		max_path = ""
		for ko_str in mod_sets[mod]:
			if mod_sets[mod][ko_str] > max_count: # if there is a tie, the first one is kept 
				max_count = mod_sets[mod][ko_str]
				max_path = ko_str.split("_")
		mod_to_ko_clean[mod] = max_path
	return mod_to_ko_clean


# Are input mods in the output?
# input: test_input_mods
# output: iterate through row, get idx on input mods, query whether they are present

def create_mod_to_kos(org_to_mod_to_kos):
	mod_to_kos = {}
	for org in org_to_mod_to_kos:
		mods = org_to_mod_to_kos[org]
	
		for mod in mods:
			if mod not in mod_to_kos:
				mod_to_kos[mod] = mods[mod]
	return mod_to_kos
	
def remove_duds(train_data, train_genomes, tnum_to_tla, org_to_mod_to_kos, n_mods):
	# remove genomes encoding < 10 modules
	keep_idx = []
	for i in range(train_data.shape[0]):
		tnum = train_genomes[i]
		org = tnum_to_tla[tnum]
		if len(org_to_mod_to_kos[org]) >= n_mods:
			keep_idx.append(i)
	train_data = train_data[keep_idx,:]
	train_genomes = list(np.array(train_genomes)[keep_idx])
	
	return train_data, train_genomes

def mod_names():
	path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_modules.txt"
	file = open(path).readlines()
	file = list(map(str.strip, file))
	
	mod_to_name = {}
	for s in file:
		if s[0] == 'D':
			tempy = s.split()
			end_idx = 0
			done = False
			for i in range(len(tempy)):
				if '[PATH' in tempy[i]:
					end_idx = i
					done = True
					mod_to_name[tempy[1]] = ' '.join(tempy[2:int(end_idx)])
					break
			if done:
				pass
			else:
				mod_to_name[tempy[1]] = ' '.join(tempy[2:])
	return mod_to_name			  	