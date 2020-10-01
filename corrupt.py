import random
import numpy as np
import torch

# For one rep of one genome
def heart_of_corruption_v1(org_to_mod_to_kos, org, n_max, n_kos_tot, all_kos, mod_to_ko_clean):
    """
    For each genome, keep the KO's in 1-10 modules. Everything else should be zeros
    
    Arguments:
    org (str) -- tla for genome (e.g.: "aha")
    n_max (int) -- the maximum number of mods to select for corrupted version of any given genome
    
    Returns:
    corrupted (np array) -- 
    """
    n_mods = random.randint(1, n_max) # going to select this many mods for corrupted genome
    keeps = random.sample(list(org_to_mod_to_kos[org].keys()), n_mods)            
    #print(org, keeps)
    idxs = []
    for mod in keeps:
        for ko in org_to_mod_to_kos[org][mod]:
        #for ko in mod_to_ko_clean[mod]:
            idxs.append(all_kos.index(ko))

    # create corrupted version of genome that only has those mods
    corrupted = np.zeros(n_kos_tot)
    for i in idxs:
        corrupted[i] = 1

    return corrupted


def heart_of_corruption_v2(org_to_mod_to_kos, org, n_max, n_kos_tot, all_kos, mod_to_ko_clean):
    """
    For each genome, remove only one KO (convert bits to zeros) -- how well can VAE restore a single module?
    
    Arguments:
    org (str) -- tla for genome (e.g.: "aha")
    n_max (int) -- the number of mods in a given genome
    
    Returns:
    corrupted (np array) -- 
    """
    keeps = random.sample(list(org_to_mod_to_kos[org].keys()), (n_max-1))            

    idxs = []
    for mod in keeps:
        for ko in org_to_mod_to_kos[org][mod]:
        #for ko in mod_to_ko_clean[mod]:
            idxs.append(all_kos.index(ko))

    # create corrupted version of genome that only has those mods
    corrupted = np.zeros(n_kos_tot)
    for i in idxs:
        corrupted[i] = 1

    return corrupted



def corrupt(train_data, train_genomes, n_corrupt, tnum_to_tla, org_to_mod_to_kos, all_kos, mod_to_ko_clean,  method):
    """
    For each genome, keep the KO's in 1-10 modules. Everything else should be zeros
    Note: creates corrupted + matching uncorrupted tensor of genomes, in that order
    Note: only genomes with >= 1 module are included in the output
    Note: uses "cleaned" modules from mod_to_ko_clean  
        I.e. most common set of KOs per module, rather than 20 variants of each mod
    
    Arguments:
    train_data (tensor) -- rows = uncorrupted genomes, columns = KOs
    train_genomes (list) -- names of genomes in train_data (e.g.: "T03060")
    n_corrupt (int) -- number of corrupted versions to make of each genome
    tnum_to_tla (dict) -- maps tnum (e.g.: "T03060") to tla (e.g.: "Red")
    method (str) -- method for performing corruption, "v1" | "v2"
    
    Returns:
    output (tensor) -- corrupted + uncorrupted genomes (each genome's two versions are concatenated in a row)
    c_train_genomes -- names of genomes in the order they appear in output
    """
        
    output = [] 
    c_train_genomes = []
    n_kos_tot = train_data.shape[1]
    
    line_counter = 0
    for i, tnum in enumerate(train_genomes):
        org = tnum_to_tla[tnum]
        n_tot_mods = len(org_to_mod_to_kos[org]) # number of modules in the genome 
        
        # needed for type v1 corruption
        n_max = min(n_tot_mods, 10) # which is smaller: the # mods or 10
        
        n_corrupted = 0
        if n_tot_mods >= 1: 
            uncorrupted = train_data[i]
            while n_corrupted < n_corrupt: 
                c_train_genomes.append(org)
                #corrupted = heart_of_corruption_v1(org_to_mod_to_kos, org, n_max, n_kos_tot, all_kos)
                corrupted = heart_of_corruption_v2(org_to_mod_to_kos, org, n_max, n_kos_tot, all_kos, mod_to_ko_clean)
                
                genome_out = np.concatenate((corrupted, uncorrupted), axis=None)
                output.append(genome_out)
                line_counter += 1
                n_corrupted += 1
            
    return torch.Tensor(np.array(output)), c_train_genomes
    















