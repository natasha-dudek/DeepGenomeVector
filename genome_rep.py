import sys, os, string, re, random

from argparse import Namespace
import numpy as np
import pandas as pd


flags = Namespace(DATA_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_autoencoder/data/')

def breakup_brackets(rxn):
    """
    # separates KOs into reactions or sub-reactions

    # A space " " between means different reactions
    # Brackets mean the same reaction 
    
    Arguments:
    rxn --  a string of KOs + reactions in a module
        E.g.: M00001 = '(K00844,K12407,K00845,K00886,K08074,K00918) (K01810,K06859,K13810,K15916)  (K00850,K16370,K21071,K00918) (K01623,K01624,K11645,K16305,K16306) K01803  ((K00134,K00150) K00927,K11389) (K01834,K15633,K15634,K15635) K01689 (K00873,K12406)'
    
    Returns:
    keeps_clean -- a list of reactions (note: some reactions may have alternative pathways, like the one above)
        E.g.: ['K00844,K12407,K00845,K00886,K08074,K00918', 'K01810,K06859,K13810,K15916', 'K00850,K16370,K21071,K00918', 'K01623,K01624,K11645,K16305,K16306', 'K01803', '(K00134,K00150) K00927,K11389', 'K01834,K15633,K15634,K15635', 'K01689', 'K00873,K12406']
    """
    keeps = []
    num_open = 0
    num_closed = 0
    seq = ""

    for i in rxn:


        # when starting a new bracket
        if i == "(" and num_open == 0:
            #print("opening",seq)
            if len(seq)> 0 and seq[0] == ")":
                seq = seq[2:]
            # catch 1+ solo KO's in between bracket sets
            # (K00036,K19243) K00033 K01783 (K01807,K01808)
            if not seq.isspace() and len(seq) > 5:  # KO#s are 6+ characters
                # get rid of misc whitespace
                seq = seq.strip()
                # if (K00033,K00033) K0# K0# (K00033,K00033)
                seq2 = seq.split()
                if len(seq2) > 1:
                    for j in seq2:
                        keeps.append(j)
                else:
                    keeps.append(seq)
            seq = ""
            num_open += 1
            #print("seq",seq, "keeps",keeps)

        # when continuing inside a bracket
        elif i == "(":
            num_open += 1        

        if i == ")":
            num_closed += 1

            # if we've suceeding in closing a full set, set everything to zero
            if num_open == num_closed != 0:
                #print("ENDING",seq+")")
                if not seq.isspace() and len(seq) > 3:
                    #print("seq.strip()", seq.strip())
                    keeps.append(seq.strip()+")")
                seq = ""
                num_open = 0
                num_closed = 0

        seq = seq + i
        #print("i", i)
    
    # catch cases where a single K0# is at the end of the pathway, like K00033 below
    # '(K13937,((K00036,K19243) (K01057,K07404))) K00033'
    if len(seq.strip()) > 5: # KO#s are 6+ characters
        seq = seq.strip()
        if seq[0] == ")":
            seq = seq[2:] # remove ") " from ") K0#"
        
        # If there is more than one trailing KO
        # 'K00036 (K01057,K07404) K01690 K01625'
        seq2 = seq.split()
        if len(seq2) > 1:
            for j in seq2:
                keeps.append(j)
        else:
            keeps.append(seq)
                        
    keeps_clean = []
    for i in keeps:
        if i[0] == "(" and "K" in i: # don't allow "--" - see M00014
            keeps_clean.append(i[1:-1])
        elif "K" in i:
            keeps_clean.append(i)
    #print(i, num_open, num_closed)
    return keeps_clean

def genome_kos(path):
	mod_file = open(path+"mod_dict.txt").readlines()
	mod_file = list(map(str.strip, mod_file))
	
	genome_file = open(path+"tnum_to_ko.txt").readlines()
	genome_file = list(map(str.strip, genome_file))
	
	# Create a dictionary of all modules that exist in KEGG db
	# keys = module ID (e.g.: 'M00001')
	# mod_dict[mod][0] = name of module (e.g.: 'Glycolysis (Embden-Meyerhof pathway), glucose > pyruvate')
	# mod_dict[mod][1] = KEGG representation of pathway (e.g.: '(K00844,K12407,K00845,K00886,K08074,K00918) (K01810,K06859,K13810,K15916)  (K00850,K16370,K21071,K00918) (K01623,K01624,K11645,K16305,K16306) K01803  ((K00134,K00150) K00927,K11389) (K01834,K15633,K15634,K15635) K01689 (K00873,K12406)')
	# mod_dict[mod][2] = lazy representation of pathway (e.g.: [['K00844', 'K12407', 'K00845', 'K00886', 'K08074', 'K00918'], ['K01810', 'K06859', 'K13810', 'K15916'], ['K00850', 'K16370', 'K21071', 'K00918'], ['K01623', 'K01624', 'K11645', 'K16305', 'K16306'], ['K01803'], ['K00134', 'K00150', 'K00927', 'K11389'], ['K01834', 'K15633', 'K15634', 'K15635'], ['K01689'], ['K00873', 'K12406']])
	
	mod_dict = {}
	for s in mod_file:
	    tempy = s.split("*")
	    mod_num = tempy[0]
	    mod_name = tempy[1]
	    rxn = tempy[2]
	    rxn_reformatted = breakup_brackets(rxn)
	    rxn_reformatted = [re.findall(r'K\d*', i) for i in rxn_reformatted]
	    # Some modules are definied in terms of other modules rather than KO's (i.e. redundant) - skip those
	    # E.g.: M00613*Anoxygenic photosynthesis in green nonsulfur bacteria*M00597 M00376
	    if len(rxn_reformatted) > 0:
	        mod_dict[mod_num] = [mod_name, rxn, rxn_reformatted]
	
	# Number of unique KO numbers in modules in the KEGG db
	ko_list = []
	for mod in mod_dict:
	    for s in mod_dict[mod][2]:
	        ko_list.extend(s)
	
	# For each genome, create list of KO numbers
	# Don't allow a KO number to be included in a genome multiple times
	genome_dict = {}
	for s in genome_file:
	    if not s: continue
	    genome = s.split("*")[0]
	    ko = s.split("*")[1]
	    if genome in genome_dict:
	        genome_dict[genome].append(ko)
	    else:
	        genome_dict[genome] = [ko]
	
	genome_dict.update({n: list(set(genome_dict[n])) for n in genome_dict.keys()})
	
	
	# convert genomes to KEGG modules
	# also keep KOs that are not part of KEGG modules
	# genome_to_mod is the representation of each genome that will be used by the autoencoder
	# keys = t number (e.g.: 'T00342')
	# values = modules + spare KOs (e.g.: M00001, M000002, K12345)
	genome_to_mod = {}
	for genome in genome_dict:
	    print(genome)
	    genome_mods = []
	    used_kos = []
	    for mod in mod_dict:
	        num_rxns = 0
	        num_reqd_rnxs = len(mod_dict[mod][2])
	        
	        #print("genome_dict[genome]",genome_dict[genome])
	        
	        for rxn in mod_dict[mod][2]:
	            #print(rxn)
	            # Find KOs that belong to the module
	            intersection = list(set(rxn) & set(genome_dict[genome]))
	            if len(intersection) > 0:
	                num_rxns += 1
	                #print("keeper")
	        # If a module is sufficiently represented in a genome, add the module to genome_mods 
	        # Add the used kos to the used_kos list
	        if ((num_reqd_rnxs - 1) <= num_rxns and num_reqd_rnxs > 3) or num_reqd_rnxs == num_rxns:
	        #if num_reqd_rnxs == num_rxns:
	            #print(genome, mod, intersection)
	            genome_mods.append(mod)
	            used_kos.extend(intersection)
	    
	    #print (len(used_kos), len(genome_mods))
	    genome_to_mod[genome] = genome_mods
	    
	    # Find all genome KOs that were not used in mods, keep those in genome_to_ko
	    unused_kos = list(set(genome_dict[genome]) - set(used_kos))
	    genome_to_mod[genome].extend(unused_kos)
	    
	list_mods = []
	for genome in genome_to_mod:
	    #print(genome_to_mod[genome])
	    list_mods.extend(genome_to_mod[genome])
	
	# Make a dict mapping modules/KOs back to genomes (i.e. KO12345 in genome A, B, C)
	num_used_mod = []
	num_free_KO = []
	mod_to_genome = {}
	num_elements = 0
	for genome in genome_to_mod:
	    for element in genome_to_mod[genome]:
	        num_elements += 1
	        # element is something like K12345 or M00001
	        if element[0][0] == "K":
	            num_free_KO.append(element)
	        elif element[0][0] == "M":
	            num_used_mod.append(element)
	        else:
	            print("ERROR", element[0])
	        
	        if element in mod_to_genome:
	            mod_to_genome[element].append(genome)
	        else:
	            mod_to_genome[element] = [genome]
	
	col_names = mod_to_genome.keys()
	
	save_dict = {}
	for genome in genome_to_mod:
	    save_dict[genome] = [1 if ko in genome_to_mod[genome] else 0 for ko in col_names]
	
	df = pd.DataFrame(save_dict).T
	df.columns = list(col_names)
	df.to_csv(flags.DATA_FP+'genome_to_mod.csv')