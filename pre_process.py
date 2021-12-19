import numpy as np
import pickle
import re
from collections import defaultdict
import torch
import random
from datetime import date

def balanced_split(f_test, final_genomes, taxid_to_tnum):
	"""
	Create a train-test split that is phylogenetically balanced at the phylum level
	
	Arguments:
	f_test (int) -- proportion allocated to test set (e.g.: 0.1 for 90% train, 10% test split)
	
	Returns:
	train_test (dict of lists) -- list of genomes (tnums) assigned to the respective keys 'train' or 'test
	"""
	# create dict mapping phyla to a list of genomes per phylum
	phylum_to_tnum = defaultdict(list)
	for taxid in final_genomes:
		#tnum = taxid_to_tnum[taxid]
		if 'k__Bacteria' not in final_genomes[taxid]: continue
		try:
			phylum = final_genomes[taxid][1]
		except IndexError:
			# some genomes are not clasified at the phylum level
			phylum = 'UNK'
		if phylum == 'Proteobacteria':
			# proteobacteria is a polyphyletic group -- use class instead
			phylum = final_genomes[taxid][2]
		tnum = taxid_to_tnum[taxid]	
		phylum_to_tnum[phylum].append(tnum)

	# do the actual split
	train_test = {'train': [], 'test': []}
	# Create split
	for phylum in phylum_to_tnum:
		num_genomes = len(phylum_to_tnum[phylum])
		
		# if few genomes, 50-50 split to training vs test set
		if 1 < num_genomes < 10:
			n_test = round(num_genomes*0.5)
			n_train = num_genomes - n_test
		# if only one genome in a phylum ---> training set
		elif num_genomes == 1:
			n_test = 0
			n_train = 1
		# otherwise do prescribed train-test split
		else:
			n_test = round(num_genomes*f_test)
			n_train = num_genomes - n_test
		
		# randomly select n_train genomes from phylum for the training set
		g_train = random.sample(phylum_to_tnum[phylum], n_train)
		g_test = []
		for genome in phylum_to_tnum[phylum]:
			if genome not in g_train:
				g_test.append(genome)
		
		train_test['train'].extend(g_train)
		train_test['test'].extend(g_test)
	
	return train_test

def genomes2include(path):
	"""
	Figure out which genomes I want to include in my dataset based on those in selected_kegg.txt
	
	Arguments: None
	
	Returns: 
	tla_to_tnum (dict) -- converts three-letter abbreviations (e.g.: eco) to t-numbers (e.g.: T04989) 
	keepers (list) -- list of t-numbers to keep (e.g.: T04989)
	"""
	
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
#		if line_counter < 4:
#			line_counter += 1
#			continue
		tla = s.split()[1]
		t_num = s.split()[2]
		tax = s.split()[3]
		   
		if "k__Bacteria" in tax:
			keepers.append(t_num)
			tla_to_tnum[tla] = t_num
	
	tnum_to_tla = {v:k for k,v in tla_to_tnum.items()}
	
	return tla_to_tnum, tnum_to_tla, keepers

def load_kos(tla_to_tnum, tnum_to_tla, tla_to_mod_to_kos, path):
	"""
	Load mapping of genome to KOs encoded
	
	Arguments:
	tla_to_tnum (dict) -- converts three-letter abbreviations (e.g.: eco) to t-numbers (e.g.: T04989) 
	tla_to_mod_to_kos (dict of dict of list) -- 
	mode () --
	
	Returns:
	tnum_to_kos (dict of lists) -- for each organism, a list of KOs encoded by the genome 
	n_kos_tot (int) -- number of KOs in tnum_to_kos
	"""

	#path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/annotations/annotations_list.txt"
	master_file = open(path+"annotations/annotations_list.txt").readlines()
	master_file = list(map(str.strip, master_file))
	
	### Create dict mapping genomes to encoded KOs
	# key = t_num 
	# ko = list of all KOs annotated in genome
	tnum_to_kos = {}
	for i in master_file:
		file = open(path+'annotations/'+i).readlines()
		file = list(map(str.strip, file))
	   
		tla = i.split("_")[0]
	   
		try:
			t_num = tla_to_tnum[tla]
		except KeyError: continue # not all will be present in keepers (e.g. "Pea")
	   
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
				   
		tnum_to_kos[t_num] = kos   
	
	org_list = list(tnum_to_kos.keys())
	for tnum in org_list:
		if tnum_to_tla[tnum] not in tla_to_mod_to_kos:
			del tnum_to_kos[tnum]
			
	# Create unique list of all KOs (all_kos) 
	all_kos = []
	for t_num in tnum_to_kos:
		all_kos.extend(tnum_to_kos[t_num])
	all_kos = list(set(all_kos))
	n_kos_tot = (len(all_kos))
	print("Total number of KOs in dataset: {}".format(n_kos_tot))

	return tnum_to_kos, n_kos_tot, all_kos

def load_mods(path):
	"""
	Load mapping of genomes to modules and KOs encoded per module

	Arguments:
	
	Returns:
	tla_to_mod_to_kos (dict of dict of list) -- three letter genome ID (e.g.: "has") to modules endoded to list of genes per module
	mod_sets (defaultdict of defaultdict of int) -- for each module, all alternative pathways and their counts 
	"""
	# Load mapping from organism (tla, e.g.: Pea) to complete modules encoded to KOs in each module
	
	# may change file name to "tla_to_mod_to_kos.pkl"
	with open(path+'tla_to_mod_to_kos.pkl', 'rb') as f:
		tla_to_mod_to_kos = pickle.load(f)
		
	# Create dict: mod_sets
	# For each module, have a counter (dict) of each unique string of KOs that can make up the module + # occurences 
	# mod_sets['M00001'] = {'K00001_K04398': 5, 'K00002_K23456': 10}
	mod_sets = defaultdict(lambda: defaultdict(int))
	for org in tla_to_mod_to_kos:
		for mod in tla_to_mod_to_kos[org]:
			ko_str = "_".join(tla_to_mod_to_kos[org][mod])
			mod_sets[mod][ko_str] += 1
	
	# # organism "lkm" module "M00083" has a K0 "K00668" that is specific to fungi
	# It does not appear in bacterial genomes (verified for this set) 
	# Its presence in tla_to_mod_to_kos will break code later on if not removed
	bad_ko = tla_to_mod_to_kos["lkm"]["M00083"].index("K00668")
	del tla_to_mod_to_kos["lkm"]["M00083"][bad_ko]
	
	return tla_to_mod_to_kos, mod_sets



def make_tensor(tla_to_mod_to_kos, tnum_to_kos, n_kos_tot, tla_to_tnum, all_kos, save=False):
	"""
	Convert tnum_to_kos dict to a tensor
	
	Arguments: 
	tla_to_mod_to_kos ()
		
	Returns:
	data (tensor) -- rows are genomes, columns are KOs 
	genome_order (list) -- in the same order as data tensor, list of genomes IDs
	"""
	
	n_genomes = len(tla_to_mod_to_kos)
	data = np.zeros(shape=(n_genomes,n_kos_tot))
	
	genome_order = []
	for i, tla in enumerate(tla_to_mod_to_kos):
		genome_order.append(tla)
		tnum = tla_to_tnum[tla]
		for j in range(n_kos_tot):
			if all_kos[j] in tnum_to_kos[tnum]:
				data[i,j] = 1
			else: pass				
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


def prep_data(list_genomes, all_kos, tnum_to_kos, mode):
	"""
	Creates a tensor for training / test data
	
	Arguments:
		list_genomes (list) -- t_nums to be included in the tensor (i.e. train_genomes or test_genomes)
		all_kos (list) -- all KOs that exist in the dataset
		tnum_to_kos (dict) -- keys = t_nums, values = list of KOs encoded by genome
		mode (str) -- used to save data to file ["test" | "train"] 
	
	Returns:
		data (np array) -- rows = genomes, columns = KOs, 1 = KO present in genome, 0 = KO absent in genome
	"""
	
	#assert (mode == "test" or mode == "train")
	
	data = np.zeros(shape=(len(list_genomes),len(all_kos)))
		
	for i, tnum in enumerate(list_genomes):
		if tnum not in tnum_to_kos:
			# annotations may not be available for all genomes
			continue
		for j, ko in enumerate(all_kos):
			if ko in tnum_to_kos[tnum]:
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

def create_mod_to_kos(tla_to_mod_to_kos):
	mod_to_kos = {}
	for org in tla_to_mod_to_kos:
		mods = tla_to_mod_to_kos[org]
	
		for mod in mods:
			if mod not in mod_to_kos:
				mod_to_kos[mod] = mods[mod]
	return mod_to_kos
	
def remove_duds(train_data, train_genomes, tnum_to_tla, tla_to_mod_to_kos, n_mods):
	# remove genomes encoding < 10 modules
	keep_idx = []
	for i in range(train_data.shape[0]):
		tnum = train_genomes[i]
		org = tnum_to_tla[tnum]
		if len(tla_to_mod_to_kos[org]) >= n_mods:
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
	
def filter(n_min, train_data, test_data, train_genomes, test_genomes):
	good_idx_train = train_data.sum(axis=1) >= n_min
	good_idx_test = test_data.sum(axis=1) >= n_min
	train_data = train_data[good_idx_train,:]
	test_data = test_data[good_idx_test,:]
	
	# to numpy for indexing, then back to list for using
	train_genomes = list(np.array(train_genomes)[good_idx_train])
	test_genomes = list(np.array(test_genomes)[good_idx_test])
	
	return train_data, test_data, train_genomes, test_genomes

def get_taxids(DATA_FP):
	# Map tnum -> taxids for all orgs in KEGG db
	dl_path = DATA_FP+'kegg_dl/'
	file = open(DATA_FP+'downloaded_infoFiles.txt').readlines()
	file = list(map(str.strip, file))
	
	tnum_to_tla = {}
	taxid_to_tnum = {}
	for s in file:
	    tla = s.split("_")[0]
	    info_file = open(dl_path+s).readlines()
	    info_file = list(map(str.strip, info_file))
	    for i in info_file:
	        if 'class="title10">Genome information' in i:
	            tnum = i.split("href='/dbget-bin/www_bget?gn:")[1].split("'")[0]
	        if '<b>Taxonomy</b></td><td>TAX:' in i:
	            taxid = i.split("https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=")[1].split('">')[0]
	    
	    taxid_to_tnum[taxid] = tnum
	    tnum_to_tla[tnum] = tla
	
	tla_to_tnum = {v:k for k,v in tnum_to_tla.items()}
	tnum_to_taxid = {v:k for k,v in taxid_to_tnum.items()} 
	
	return taxid_to_tnum, tnum_to_tla

def taxid_to_tax(DATA_FP, taxid_to_tnum, date):
	
	file = open(DATA_FP+'ncbi_lineages_'+date+'.csv').readlines()
	file = map(str.strip, file) 
	
	tnum_to_tax = {}
	
	for i in file:
	    lin = i.split(',')
	    taxid = lin[0]
	    
	    if taxid in taxid_to_tnum:
	        tnum = taxid_to_tnum[taxid]
	        tnum_to_tax[tnum] = lin[1:8] # domain - species
	        
	return tnum_to_tax
	
def phylogenetic_thin(tnum_to_tax):
	"""
	Arguments:
	tnum_to_tax (dict of lists) -- maps tnum to taxonomy in form of [Domain, Phylum, Class, ..., Species]
	
	Returns:
	final_genomes (defaultdict) -- keys are genomes to keep for analysis
								   values are a list of [Domain, Phylum, Class, ..., Species]
	"""
	
	# Make dictionary mapping species to genomes
	spp_to_genomes = defaultdict(list)
	for tnum in tnum_to_tax:
	    spp = tnum_to_tax[tnum][6]
	    spp_to_genomes[spp].append(tnum)

	# Randomly select only one representative genome per species and store in final_genomes dict
	# Simultaneously ensure that for genomes identified to the level of species, 
	####### there are no more than 5 representatives per genus
	####### take all genomes not classified at the genus level
	final_genomes = defaultdict(list)
	spp_counter = 0
	num_thresh = 5 # number of spp per genus
	num_per_genus = defaultdict(list)
	
	for spp in spp_to_genomes:
	    # randomly select one representative per spp to keep
	    tnum_keep = random.choice(spp_to_genomes[spp]) 
	    # Now check that the genus from which this species derives is not full
	    genus = tnum_to_tax[tnum_keep][5]
	        
	    num_per_genus[genus].append(tnum_keep)
	    
	    if len(genus) < 1:
	        final_genomes[tnum_keep] = tnum_to_tax[tnum_keep]    
	    
	    elif len(num_per_genus[genus]) <= num_thresh:
	        final_genomes[tnum_keep] = tnum_to_tax[tnum_keep]
	
	return final_genomes
	
def thin2():
	#master_file = open('/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/downloaded_3LA.txt').readlines()
	master_file = open('/Users/natasha/Desktop/kegg_mar2021/downloaded_infoFiles.txt').readlines()
	master_file = map(str.strip, master_file)
	
	tax_path = open("/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/ncbi_lineages_2020-05-04.csv").readlines()
	tax_path = map(str.strip, tax_path) 
	
	tla_to_info = {}
	
	for i in master_file:
		file = open("/Users/natasha/Desktop/kegg_mar2021/kegg_dl/"+i).readlines()
		#file = open("/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_dl/"+i).readlines()
		file = map(str.strip, file)
		
		threeLA = i.split("_")[0]
		
		for s in file:
			if "T number" in s:
				t_number = s.split("href='/dbget-bin/www_bget?gn:")[1].split("'>")[0]
			elif "<b>Taxonomy</b></td><td>TAX:" in s:
				taxid = s.split("https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=")[1].split('">')[0]
		
		tla_to_info[threeLA] = [t_number, taxid]
	
	# create inverted dictionary
	# key = taxid
	# value = three letter identifier
	taxid_to_tla = {}
	for tla in tla_to_info:
		taxid = tla_to_info[tla][1]
		taxid_to_tla[taxid] = tla
		
	
	############
	# Create mapping from taxid to taxonomic classification
	# Only retain Bacteria
	# E.g.: 6473826429: p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Anaerocolumna;s__Anaerocolumna sp. CBA3638
	############
	
	tax_dict = {}
	
	for s in tax_path:
		tempy = s.split(",")
		taxid = tempy[0]
		domain = tempy[1]
		phylum = tempy[2]
		classS = tempy[3]
		order = tempy[4]
		family = tempy[5]
		genus = tempy[6]
		species = tempy[7]
		
	#	if domain != "Bacteria":
	#		continue
			
		if len(classS) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum
		elif len(order) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS	
		elif len(family) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order
		elif len(genus) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order+";f__"+family
		elif len(species) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order+";f__"+family+";g__"+genus
		else:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order+";f__"+family+";g__"+genus+";s__"+species
	
	############
	# Randomly select only one representative genome per species and store in final_genomes dict
	# Do not take more than num_thresh species per genus
	############
	
	# for each genome, have lineage
	# E.g.: 'zmm': 'p__Proteobacteria;c__Alphaproteobacteria;o__Sphingomonadales;f__Sphingomonadaceae;g__Zymomonas;s__Zymomonas mobilis'
	tla_to_lineage = {}
	badies = []
	for s in tla_to_info: # s is a genome
		try:
			taxid = tla_to_info[s][1]
			lineage = tax_dict[taxid]
			tla_to_lineage[taxid] = lineage
		except KeyError: pass # a few may be missing
	
	# Group genomes by taxonomic ID
	# E.g.: 'p__Proteobacteria;c__Alphaproteobacteria;o__Sphingomonadales;f__Sphingomonadaceae;g__Zymomonas;s__Zymomonas mobilis': [zmm, zma, zmd]
	spp_to_genome = {}
	
	for s in tla_to_lineage:
		if tla_to_lineage[s] in spp_to_genome:
			spp_to_genome[tla_to_lineage[s]].append(s)
		else:
			spp_to_genome[tla_to_lineage[s]] = [s]
	
	# Now select final genomes
	final_genomes = {}
	
	# number of spp. representatives per genus
	num_thresh = 5
	
	# For genomes classified to the level of species
	# Make sure that no more than num_thresh spp per genus are included in final output
	num_per_genus = {}
	random.seed(0)
	spp_counter = 0
	
	for s in spp_to_genome:
		if ";s__" in s:
			genus = s.split(";")[5]
			
			if genus in num_per_genus:
				num_per_genus[genus] += 1
			else:
				num_per_genus[genus] = 1
			
			# No more than num_thresh +1 spp per genus
			if num_per_genus[genus] <= num_thresh + 1:
				selected_genome = random.choice(spp_to_genome[s])
				spp_counter += 1
				
				if "Babela massiliensis" in s: # Phylum for this spp is undefined, should be TM6
					s = 'k__Bacteria;p__TM6;c__Candidatus_Babeliae;o__Candidatus_Babeliales;f__Candidatus_Babeliaceae;g__Candidatus_Babela;s__Candidatus_Babela_massiliensis'
				
				final_genomes[selected_genome] = s
	
	
	############
	# Now lets add in some of the more novel bacterial diversity (not classified to the level of species)
	# E.g.: Many CP do not have defined species at present, but don't want to ignore that the phylum exists 
	# Only allow 50 randoms per phylum to be added
	# Do not include genomes not classified at the phylum level
	############
	
	# invert spp_to_genome
	# keys: genome -- something like GCA_010384365.1_ASM1038436v1
	# values: taxonomic path -- something like p__Proteobacteria;c__Gammaproteobacteria;o__Vibrionales
	inv_spp_to_genome = {}
	for k,v in spp_to_genome.items():
		for genome in v:
			inv_spp_to_genome[genome] = k
	
	unk_per_phylum = {}	
	unk_count = 0
	for s in spp_to_genome: # s is something like 'p__Proteobacteria;c__Gammaproteobacteria;o__Oceanospirillales'
		
		phylum = s.split(";")[1]
		
		if ";s__" not in s and len(phylum) > 3: # get rid of anything not classified at phylum level p__		
	
			if phylum in unk_per_phylum:
				unk_per_phylum[phylum].extend(spp_to_genome[s])
			else:
				unk_per_phylum[phylum] = spp_to_genome[s]
		
	max_per_phy = 50
	for phylum in unk_per_phylum:					
		# if there are more than 50 genomes in the phylum that have not been classified down to the level of spp
		# only take 50
			
		if len(unk_per_phylum[phylum]) > max_per_phy:
			selected_genomes = random.sample(unk_per_phylum[phylum], k=max_per_phy)	# random.sample does not use replacement	
		else:
			selected_genomes = unk_per_phylum[phylum]
			
		for genome in selected_genomes:
			# genome looks something like GCA_001653795.1_ASM165379v1
			final_genomes[genome] = inv_spp_to_genome[genome]
			unk_count += 1
	return final_genomes



def thin3():
	#master_file = open('/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/downloaded_3LA.txt').readlines()
	master_file = open('/Users/natasha/Desktop/kegg_mar2021/downloaded_infoFiles.txt').readlines()
	master_file = map(str.strip, master_file)
	
	tax_path = open("/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/ncbi_lineages_2020-05-04.csv").readlines()
	tax_path = map(str.strip, tax_path) 
	
	tla_to_info = {}
	
	for i in master_file:
		file = open("/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_dl/"+i).readlines()
		file = map(str.strip, file)
		
		threeLA = i.split("_")[0]
		
		for s in file:
			if "T number" in s:
				t_number = s.split("href='/dbget-bin/www_bget?gn:")[1].split("'>")[0]
			elif "<b>Taxonomy</b></td><td>TAX:" in s:
				taxid = s.split("https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=")[1].split('">')[0]
		
		tla_to_info[threeLA] = [t_number, taxid]
	
	# create inverted dictionary
	# key = taxid
	# value = three letter identifier
	taxid_to_tla = {}
	for tla in tla_to_info:
		taxid = tla_to_info[tla][1]
		taxid_to_tla[taxid] = tla
		
	
	############
	# Create mapping from taxid to taxonomic classification
	# Only retain Bacteria
	# E.g.: 6473826429: p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Anaerocolumna;s__Anaerocolumna sp. CBA3638
	############
	
	tax_dict = {}
	
	for s in tax_path:
		tempy = s.split(",")
		taxid = tempy[0]
		domain = tempy[1]
		phylum = tempy[2]
		classS = tempy[3]
		order = tempy[4]
		family = tempy[5]
		genus = tempy[6]
		species = tempy[7]
		
	#	if domain != "Bacteria":
	#		continue
			
		if len(classS) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum
		elif len(order) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS	
		elif len(family) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order
		elif len(genus) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order+";f__"+family
		elif len(species) < 1:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order+";f__"+family+";g__"+genus
		else:
			tax_dict[taxid] = "k__"+domain+";p__"+phylum+";c__"+classS+";o__"+order+";f__"+family+";g__"+genus+";s__"+species
	
	############
	# Randomly select only one representative genome per species and store in final_genomes dict
	# Do not take more than num_thresh species per genus
	############
	
	# for each genome, have lineage
	# E.g.: 'zmm': 'p__Proteobacteria;c__Alphaproteobacteria;o__Sphingomonadales;f__Sphingomonadaceae;g__Zymomonas;s__Zymomonas mobilis'
	tla_to_lineage = {}
	badies = []
	for s in tla_to_info: # s is a genome
		taxid = tla_to_info[s][1]
		lineage = tax_dict[taxid]
		tla_to_lineage[taxid] = lineage
	
	
	# Group genomes by taxonomic ID
	# E.g.: 'p__Proteobacteria;c__Alphaproteobacteria;o__Sphingomonadales;f__Sphingomonadaceae;g__Zymomonas;s__Zymomonas mobilis': [zmm, zma, zmd]
	spp_to_genome = {}
	
	for s in tla_to_lineage:
		if tla_to_lineage[s] in spp_to_genome:
			spp_to_genome[tla_to_lineage[s]].append(s)
		else:
			spp_to_genome[tla_to_lineage[s]] = [s]
	
	# Now select final genomes
	final_genomes = {}
	
	# number of spp. representatives per genus
	num_thresh = 5
	
	# For genomes classified to the level of species
	# Make sure that no more than num_thresh spp per genus are included in final output
	num_per_genus = {}
	random.seed(0) # added after creating dataset and training final model...
	spp_counter = 0
	
	for s in spp_to_genome:
		if ";s__" in s:
			genus = s.split(";")[5]
			
			if genus in num_per_genus:
				num_per_genus[genus] += 1
			else:
				num_per_genus[genus] = 1
			
			# No more than num_thresh +1 spp per genus
			if num_per_genus[genus] <= num_thresh + 1:
				selected_genome = random.choice(spp_to_genome[s])
				spp_counter += 1
				
				if "Babela massiliensis" in s: # Phylum for this spp is undefined, should be TM6
					s = 'p__TM6;c__Candidatus_Babeliae;o__Candidatus_Babeliales;f__Candidatus_Babeliaceae;g__Candidatus_Babela;s__Candidatus_Babela_massiliensis'
				
				final_genomes[selected_genome] = s
	
	
	############
	# Now lets add in some of the more novel bacterial diversity (not classified to the level of species)
	# E.g.: Many CP do not have defined species at present, but don't want to ignore that the phylum exists 
	# Only allow 50 randoms per phylum to be added
	# Do not include genomes not classified at the phylum level
	############
	
	# invert spp_to_genome
	# keys: genome -- something like GCA_010384365.1_ASM1038436v1
	# values: taxonomic path -- something like p__Proteobacteria;c__Gammaproteobacteria;o__Vibrionales
	inv_spp_to_genome = {}
	for k,v in spp_to_genome.items():
		for genome in v:
			inv_spp_to_genome[genome] = k
	
	unk_per_phylum = {}	
	unk_count = 0
	for s in spp_to_genome: # s is something like 'p__Proteobacteria;c__Gammaproteobacteria;o__Oceanospirillales'
		
		phylum = s.split(";")[1]
		
		if ";s__" not in s and len(phylum) > 3: # get rid of anything not classified at phylum level p__		
	
			if phylum in unk_per_phylum:
				unk_per_phylum[phylum].extend(spp_to_genome[s])
			else:
				unk_per_phylum[phylum] = spp_to_genome[s]
		
	max_per_phy = 50
	for phylum in unk_per_phylum:					
		# if there are more than 50 genomes in the phylum that have not been classified down to the level of spp
		# only take 50
			
		if len(unk_per_phylum[phylum]) > max_per_phy:
			selected_genomes = random.sample(unk_per_phylum[phylum], k=max_per_phy)	# random.sample does not use replacement	
		else:
			selected_genomes = unk_per_phylum[phylum]
			
		for genome in selected_genomes:
			# genome looks something like GCA_001653795.1_ASM165379v1
			final_genomes[genome] = inv_spp_to_genome[genome]
			unk_count += 1
		#unk_count1 += len(selected_genomes)		
	
	bacteria = 0
	archaea = 0
	eukaryota = 0
	for i in final_genomes:
		if "Bacteria" in final_genomes[i]:
			bacteria += 1
		elif "Archaea" in final_genomes[i]:
			archaea += 1
		elif "Euk" in final_genomes[i]:
			eukaryota += 1
	
	#print("Number of representatives at species level",spp_counter)
	#print("The number of unclassified genomes is",unk_count)
	#print("Total number of genomes is", spp_counter+unk_count) 
	#print(("There are %s bacteria, %s archaea, and %s eukaryotes") % (bacteria, archaea, eukaryota))
	
	#for i in final_genomes:
	#	print (i, taxid_to_tla[i], tla_to_info[taxid_to_tla[i]][0], final_genomes[i].replace(" ", "_")+"*")
	return final_genomes, taxid_to_tla, tla_to_info

def get_tax():
	# Map tnum -> taxids for all orgs in KEGG db
	dl_path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/'
	file = open(dl_path+'downloaded_3LA.txt').readlines()
	file = list(map(str.strip, file))
	
	tnum_to_tla = {}
	taxid_to_tnum = {}
	for s in file:
	    tla = s.split("_")[0]
	    info_file = open(dl_path+'kegg_dl/'+s).readlines()
	    info_file = list(map(str.strip, info_file))
	    for i in info_file:
	        if 'class="title10">Genome information' in i:
	            tnum = i.split("href='/dbget-bin/www_bget?gn:")[1].split("'")[0]
	        if '<b>Taxonomy</b></td><td>TAX:' in i:
	            taxid = i.split("https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=")[1].split('">')[0]
	
	    taxid_to_tnum[taxid] = tnum
	    tnum_to_tla[tnum] = tla
	
	tnum_to_taxid = {v:k for k,v in taxid_to_tnum.items()} 
	
	path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/ncbi_lineages_2020-05-04.csv'
	file = open(path).readlines()
	file = map(str.strip, file) 
	
	tnum_to_tax = {}
	
	for i in file:
	    lin = i.split(',')
	    taxid = lin[0]
	
	    if taxid in taxid_to_tnum:
	        tnum = taxid_to_tnum[taxid]
	        tnum_to_tax[tnum] = lin[1:8] # domain - species
	return (tnum_to_taxid, tnum_to_tax)	