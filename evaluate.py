import sklearn as sk
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import random
from copy import deepcopy
import numpy as np
import pickle
import torch
import random
import re
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

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
    print("largest_tla",largest_tla)    # row index of smallest genome in train set
    start = c_train_genomes.index(largest_tla) # hed = Hoaglandella endobia, Gammaproteobacteria
    largest_uncorrupted = corrupted_train[start,n_features:]

    # Create baseline for test set
    baseline5 = torch.Tensor(np.tile(largest_uncorrupted, (corrupted_test.shape[0], 1)))
    
    return baseline5.long(), largest_tla

def genome_count_vs_f1(train_tax_dict, test_tax_dict):
    test_phyla = {}
    for tla in test_tax_dict:
        phylum = test_tax_dict[tla][1]
        if phylum == "Proteobacteria":
            phylum = test_tax_dict[tla][1]
        if phylum not in test_phyla:
            test_phyla[phylum] = []

    train_phyla = {}
    for tla in train_tax_dict:
        phylum = train_tax_dict[tla][1]
        if phylum == "Proteobacteria":
            phylum = train_tax_dict[tla][1]
        if phylum not in train_phyla:
            train_phyla[phylum] = 1
        else:
            train_phyla[phylum] += 1

    for f1 in f1s:
        idx = f1s.index(f1)
        tla = c_test_genomes[idx]
        phylum = test_tax_dict[tla][1]
        if phylum == "Proteobacteria":
            phylum = test_tax_dict[tla][1]
        test_phyla[phylum].append(f1)

    phylum_f1s = [np.median(test_phyla[i]) for i in test_phyla]
    phylum_count = [train_phyla[i] for i in test_phyla]

    plt.scatter(phylum_count, phylum_f1s)
    plt.xlabel("Number of genomes in train set")
    plt.ylabel("F1 score on test set")
    plt.xscale('log')
    

def compare_in_n_out(binary_pred, corrupted):
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
    #print(count_hund, len(perc_out), count_hund/len(perc_out)*100)
    
    print("There are",count_hund,"instance of inputs being 100% present in output")
    print("This is out of",total,"instances or",str(round(count_hund/total*100, 2))+"% of cases")
    print("There are",count_ninety,"instance of inputs being >=90% present in output ("+str(round(count_ninety/total*100, 2))+"%)")    
    return fig, out


def test_f1s(uncorrupted, binary_pred):
    f1s = []
    for i in range(0,len(binary_pred)):
        f1 = sk.metrics.f1_score(uncorrupted[i], binary_pred[i], zero_division=0)
        f1s.append(f1)
    
    print("median F1 score:",np.median(f1s), "min", min(f1s), "max", max(f1s))
    
    fig = fig, ax = plt.subplots()    
    plt.hist(f1s)
    plt.xlabel('F1 score')
    plt.ylabel('Count')
    return f1s, fig


def f1s_per_phylum(train_tax_dict, test_tax_dict, c_test_genomes, f1s):
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

def train_out(train_input_mods):
    train_out = defaultdict(int)
    for genome in train_input_mods:
        for mod in genome:
            train_out[mod] += 1
    return train_out

def plot_train_count_hist(train_out):
    fig = fig, ax = plt.subplots()  
    plt.hist(train_out.values())
    plt.xlabel('# times mods are used in a corrupted genome')
    plt.ylabel('Count')    
    return fig

def plot_mod_count_vs_f1(test_input_mods, c_test_genomes, train_input_mods, f1s):


    # For each mod, for each time it occurs in a genome, append F1 score of genome reconstruction
    out = defaultdict(lambda: [])
    for idx,i in enumerate(test_input_mods):
        corruption_f1 = f1s[idx]
        for mod in i:
            out[mod].append(corruption_f1)

    # get median, median absolute deviation, list of mods                
#    mad = [stats.median_absolute_deviation(out[i]) for i in out]
#    median = [np.median(out[i]) for i in out]
#    mod_list = [i for i in out]
    
    mad = []
    median = []
    mod_list = []
    for i in out:
        mad.append(stats.median_absolute_deviation(out[i]))
        median.append(np.median(out[i]))
        mod_list.append(i for i in out)
    
    # sort lists in order of decreasing median F1 score
    median, mad, mod_list = zip(*sorted(zip(median, mad, mod_list), reverse=True))
    mod_num = [i for i in out.keys()]
    print("min median",min(median),"max median",max(median))
        
    # For each mod, what type of process is it involved in?
    process_to_mod = {}
    path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_modules.txt"
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

    # plot 
        
    # palette of 50 random colours
    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
    cmap = get_cmap(len(process_to_mod))    
    
    fig = fig, ax = plt.subplots(figsize=(19,5)) 
#    bars = plt.bar(mod_num, median, yerr=mad) #, color='b')
#    ax.set_xlim(0.5,len(mod_num)-0.5)
#    # redo colouring
#    done = defaultdict(tuple)
#    for i, bar in enumerate(bars):
#        mod = mod_num[i]
#        proc = mod_to_proc[mod]
#        if proc in done:
#            bar.set_color(done[proc])
#        else:
#            #new_colour = cmap(i)
#            new_colour = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
#            #new_colour = cmap(i)
#            done[proc] = new_colour
#            bar.set_color(done[proc])

#    done = defaultdict()
#    for proc in process_to_mod:
#        new_colour = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
#        for mod in process_to_mod[proc]:
#            try:
#                mod_idx = mod_num.index(mod)
#                ax.bar(mod_num[mod_idx], median[mod_idx], yerr=mad[mod_idx], color=new_colour)
#            except ValueError: continue # not all mods are in the test set        

    done = defaultdict()
    for i, mod in enumerate(mod_num):
        proc = mod_to_proc[mod]
        if proc in done:
            new_colour = done[proc]
        else:
            new_colour = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            done[proc] = new_colour
        
        ax.bar(mod_num[i], median[i], yerr=mad[i], color=new_colour)
    
    


    
    ax.legend(done.keys())
            
    plt.xlabel('Module')
    plt.ylabel('Median F1 score')
    plt.xticks([])
    plt.tight_layout()
    #plt.xticks(rotation=90)
    return fig, done


def plot_mod_count_vs_f1_v2(test_input_mods, f1s, train_out):
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
    

def plot_metab_pathway_f1(process_to_mod, test_input_mods, f1s):
    
    # create dict mapping mods the median F1 score in test set
    # Ex: 'M00001': [0.98, 0.89, 0.76]
    out = defaultdict(lambda: [])
    for idx,i in enumerate(test_input_mods):
        corruption_f1 = f1s[idx]
        for mod in i:
            out[mod].append(corruption_f1)
    
    # Create list of median F1 per module in entire test ds
    # Also keep track of the order in which mods are passed through (mod_num)
    median = []
    mod_num = []
    for i in out:
        median.append(np.median(out[i]))
        mod_num.append(i)

    # create dict mapping processes to a list of F1 scores for mods in the process
    # Ex: 'Carbohydrate metabolism': [0.89, 0.67, 0.38]
    proc_f1s = defaultdict(list)
    for proc in process_to_mod:
        for mod in process_to_mod[proc]:
            try:
                mod_idx = mod_num.index(mod)
                proc_f1s[proc].append(median[mod_idx])
            except ValueError: continue
    
    # for each process, calculate median, median F1 score
    # then sort in order of descending F1 score
    # this will be used to order processed on the x-axis        
    list_f1s_per_proc = [] 
    list_medians_per_proc = []
    list_procs = []
    for i in proc_f1s:
        list_procs.append(i)
        list_f1s_per_proc.append(proc_f1s[i]) # use for scatter on boxplot
        list_medians_per_proc.append(np.median(proc_f1s[i])) # use to order procs on x-axis
    list_medians_per_proc, list_f1s_per_proc, list_procs = zip(*sorted(zip(list_medians_per_proc, list_f1s_per_proc, list_procs), reverse=True))
    
    # plot boxplot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])        
    bp = ax.boxplot(list_f1s_per_proc, showfliers=False)
    
    for i, proc in enumerate(list_procs):
        # add scatter on x-axis
        x = np.random.normal(i+1, 0.04, size=len(list_f1s_per_proc[i]))
        plt.plot(x, list_f1s_per_proc[i], 'r.', alpha=0.2)
    
    plt.xticks([i+1 for i in range(len(list_procs))], [proc for proc in list_procs], rotation=90)
    plt.ylabel('Median F1 score per module')        
    
    return fig

def map_proc_mod1():
    process_to_mod = {}
    path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_modules.txt"
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

def plot_mod_vs_f1(test_input_mods, f1s):

    # For each mod, for each time it occurs in a genome, append F1 score of genome reconstruction
    out = defaultdict(lambda: [])
    for idx,i in enumerate(test_input_mods):
        corruption_f1 = f1s[idx]
        for mod in i:
            out[mod].append(corruption_f1)

    # Create list of median F1 per mod
    # Also keep track of the order in which mods are passed through (mod_list)
    median = []
    mod_list = []
    for i in out:
        median.append(np.median(out[i]))
        mod_list.append(i)
    
    # sort lists in descending order of F1 score
    median, mod_list = zip(*sorted(zip(median, mod_list), reverse=True))


    fig = plt.figure(figsize=(20,5))
    ax = fig.add_axes([0,0,1,1])
    
    bp = ax.boxplot([out[i] for i in mod_list], showfliers=False)
    
    # add true data points
    for i, mod in enumerate(mod_list):
        y = out[mod]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.05)    
    
    #bp = ax.boxplot([out[i] for i in out], showfliers=False)
    bp = ax.boxplot([out[i] for i in mod_list], showfliers=False)
    
    plt.xlabel('Module')
    plt.ylabel('F1 score')
    plt.xticks([])
    
    return fig

def map_subproc_mod():
    subprocess_to_mod = {}
    path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_modules.txt"
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


def plot_mod_vs_f1_v2(test_input_mods, f1s):

    # For each mod, for each time it occurs in a genome, append F1 score of genome reconstruction
    out = defaultdict(lambda: [])
    for idx,i in enumerate(test_input_mods):
        corruption_f1 = f1s[idx]
        for mod in i:
            out[mod].append(corruption_f1)

    # Create list of median F1 per mod
    # Also keep track of the order in which mods are passed through (mod_list)
    median = []
    mod_list = []
    for i in out:
        median.append(np.median(out[i]))
        mod_list.append(i)
    
    # sort lists in descending order of F1 score
    median, mod_list = zip(*sorted(zip(median, mod_list), reverse=True))


    fig = plt.figure(figsize=(20,5))
    ax = fig.add_axes([0,0,1,1])
    
    bp = ax.boxplot([out[i] for i in mod_list], showfliers=False)
    
    # add true data points
    for i, mod in enumerate(mod_list):
        y = out[mod]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.05)    
    
    #bp = ax.boxplot([out[i] for i in out], showfliers=False)
    bp = ax.boxplot([out[i] for i in mod_list], showfliers=False)
    
    plt.xlabel('Module')
    plt.ylabel('F1 score')
    plt.xticks([])
    
    return fig

def plot_metab_pathway_f1_v2(process_to_mod, mod_to_kos, all_kos, ko_f1s, figsize):
    proc_to_ko_F1s = defaultdict(list)
    for proc in process_to_mod:
        for mod in process_to_mod[proc]:
            try:
                kos = mod_to_kos[mod]
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
        
    list_medians, list_f1s, list_procs = zip(*sorted(zip(list_medians, list_f1s, list_procs), reverse=True))
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    
    for i, proc in enumerate(list_procs):
        # add scatter on x-axis
        x = np.random.normal(i+1, 0.04, size=len(list_f1s[i]))
        plt.plot(x, list_f1s[i], 'r.', alpha=0.2)
        
    bp = ax.boxplot(list_f1s, showfliers=False)
    
    plt.xticks([i+1 for i in range(len(list_procs))], [proc for proc in list_procs], rotation=90)
    plt.ylabel('F1 score')
    
    return fig


def plot_metab_pathway_f1_v2_horizontal(process_to_mod, mod_to_kos, all_kos, ko_f1s, figsize):
    proc_to_ko_F1s = defaultdict(list)
    for proc in process_to_mod:
        for mod in process_to_mod[proc]:
            try:
                kos = mod_to_kos[mod]
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


def new_genome_random(mod_to_kos, model, all_kos, save_to, BASE_DIR):
    
    with open(BASE_DIR+'seq_dict.pkl', 'rb') as handle:
        seq_dict = pickle.load(handle)
    
    my_corrupted = torch.zeros(len(all_kos))
    
    # Pick 10 random modules as input
    n_mods = 10
    keeps = random.sample(list(mod_to_kos.keys()), n_mods)
    
    # Get the genes for those modules
    idxs = []
    for mod in keeps:
        for ko in mod_to_kos[mod]:
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
    n_gen (int) -- number of genomes to generate
    n_mods (int) -- number of modules to use as input
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

def plot_dist(generated, all_kos, mod_to_kos, model, test_data, idx=None):
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.decomposition import PCA
    
#    n_gen = 100 #test_data.shape[0] # number of fake genomes to generate
#    generated = torch.zeros(n_gen, len(all_kos))
#    
#    for i in range(n_gen):
#        
#        my_corrupted = torch.zeros(len(all_kos))
#        
#        # Pick 10 random modules as input
#        n_mods = 10
#        keeps = random.sample(list(mod_to_kos.keys()), n_mods)
#    
#        # Get the genes for those modules
#        idxs = []
#        for mod in keeps:
#            for ko in mod_to_kos[mod]:
#                idxs.append(all_kos.index(ko))
#        
#        # Turn them on in my vector
#        my_corrupted[idxs] = 1
#        
#        # Make a predicted genome
#        with torch.no_grad():
#            my_pred = model.forward(my_corrupted)[0].detach()
#        
#        my_binary_pred = eval_binarize(my_pred.reshape(1, -1), 0.5)
#        
#        # get indices that are turned on in the prediction
#        on_idx = [i[1] for i in (my_binary_pred == 1).nonzero().tolist()]
#        my_corrupted[on_idx] = 1
#        
#        generated[i] = my_corrupted
    
    n_gen = generated.shape[0]
    
    # concatenate real and fake genomes
    concated = torch.cat((torch.Tensor(test_data), generated), 0).numpy()
    #concated = torch.cat((concated, torch.Tensor(train_data)), 0).numpy()
    
    test_data_labels = ['test' for i in range(test_data.shape[0])]
    generated_labels = ['generated' for i in range(n_gen)]
    #train_data_labels = ['train' for i in range(train_data.shape[0])]
    
    df = pd.DataFrame(concated)
    
    jac_sim = 1 - pairwise_distances(df, metric = "hamming")
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(jac_sim)
    
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
                 
    labels = test_data_labels + generated_labels
    #labels = test_data_labels + generated_labels + train_data_labels
    
    labels_df = pd.Series( (v for v in labels))
    
    finalDf = pd.concat([principalDf, labels_df], axis = 1)
    
    var_one = pca.explained_variance_[0]
    var_two = pca.explained_variance_[1]
    
    
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
    
    Arguments:
        generated_genomes (tensor) -- generated genomes 
        test_data (np.ndarray) -- test genomes, rows = genomes, cols = genes
        test_genomes (list) -- genome IDs in same order as test_data 
        
    Returns:
        df (DataFrame) -- contains genome vectors for real (test) + generated genomes
    """
    
    from sklearn.metrics.pairwise import pairwise_distances
    
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
    save_to = BASE_DIR+"phylip_in.txt"
    phylum_dict = {}
    with open(save_to, 'w') as handle:
        handle.write("     "+str(df.shape[0])+"     "+str(df.shape[1])+'\n')
        
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
            handle.write(new_id+"     "+''.join([str(int(i)) for i in chars])+'\n')    
    
    return phylum_dict
    
def get_phyla_colours():
    """
    Returns pre-defined dict mapping phyla in test set to a unique colour (rbg)
    
    Arguments: None
    
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
    save_to = BASE_DIR+"vae_dendro_colours_real.txt"

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
    
    
    
    print(label_legend)
    print(colour_legend)
    
    
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
    save_to = BASE_DIR+"vae_dendro_colours_generated.txt"

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

def bio_insights_fig(test_phyla, subprocess_to_mod, all_kos, ko_f1s, mod_to_kos):
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
                kos = mod_to_kos[mod]
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

def pie_chart_reconstructions(f1s, c_test_genomes, tns, fps, fns, tps, uncorrupted, corrupted):

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
    
    def plot(data, perc, i, posn):
        """
        Plots donut plots
        """
        if posn == 2:
            #[tp_best, tn_best, fn_best, fp_best]
            labels = ['TP: '+str(data[0])+" ("+str(perc[0])+"%)", 'TN: '+str(data[1])+" ("+str(perc[1])+"%)", 'FN: '+str(data[2])+" ("+str(perc[2])+"%)", 'FP: '+str(data[3])+" ("+str(perc[3])+"%)"]
            colors = ["green", "blue", "yellow", "red"]
        else:
            colors = ["green", "blue"]
            labels = ['On: '+str(data[0])+" ("+str(perc[0])+"%)",'Off: '+str(data[1])+" ("+str(perc[1])+"%)",]
            #labels = ['On: '+str(data[3])+" ("+str(perc[3])+"%)",'Off: '+str(data[0])+" ("+str(perc[0])+"%)",]
        
        wedges, texts = axs[i,posn].pie(data, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
    
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")
    
        for s, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
        
            axs[i,posn].annotate(labels[s], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)
    
    fig, axs = plt.subplots(3, 3, figsize=(15,6))
    
    for i in range(num_charts):        
        
        # GENERATED
        posn = 2 # which subplot
        orig_idx = keeps_idx[i]
        f1_best = f1s[orig_idx]
        tn_best = tns[orig_idx]
        fp_best = fps[orig_idx]
        fn_best = fns[orig_idx]
        tp_best = tps[orig_idx]
        data = [tp_best, tn_best, fn_best, fp_best]
        total = sum(data)
        perc = [round(tp_best/total*100,2), round(tn_best/total*100,2), round(fn_best/total*100,2), round(fp_best/total*100,2)]
        plot(data, perc, i, posn)
    
        # CORRUPTED
        posn = 1
        tp_best = int(torch.sum(corrupted[orig_idx]))
        tn_best = int(corrupted.shape[1] - torch.sum(corrupted[orig_idx]))
        data = [tp_best, tn_best]    
        total = sum(data)
        perc = [round(tp_best/total*100,2), round(tn_best/total*100,2)]
        plot(data, perc, i, posn)    
    
        # UNCORRUPTED
        posn = 0
        tp_best = int(torch.sum(uncorrupted[orig_idx]))
        tn_best = int(uncorrupted.shape[1] - torch.sum(uncorrupted[orig_idx]))
        data = [tp_best, tn_best]    
        total = sum(data)
        perc = [round(tp_best/total*100,2), round(tn_best/total*100,2)]
        plot(data, perc, i, posn)    
        
        axs[0, 0].set_title("Uncorrupted")
        axs[0, 1].set_title("Corrupted")
        axs[0, 2].set_title("Reconstructed")
            
        print(c_test_genomes[orig_idx],"F1: "+str(f1_best))
        print()
        
    plt.tight_layout()
    
    return fig
    

def plot_reconstruction_barh(f1s, c_test_genomes, tns, fps, fns, tps, uncorrupted, corrupted):
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
    
    fig, axs = plt.subplots(3, 1, figsize=(15,6))
    
    for i in range(num_charts):
    
        # GENERATED
        posn = 2 # which subplot
        orig_idx = keeps_idx[i]
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
        
        colors = ['b', 'g', 'r', '#ffff00']
        labels = ["TN", "TP", "FP", "FN"]
        
        barWidth = 1
        lefts = 0
        for bars, col, label in zip([bars1, bars2, bars3, bars4], colors, labels):
            if i == 0: # add labels
                axs[i].barh(r, bars, left=lefts, color=col, edgecolor='black', height=barWidth, label=label)
                lefts += bars
            else:
                axs[i].barh(r, bars, left=lefts, color=col, edgecolor='black', height=barWidth)
                lefts += bars
        if i == 0:
            axs[0].legend()
            print("labels", labels)
        axs[i].set_xlim([0,100])
        axs[i].set_ylim(-0.5, len(bars) - 0.5)
        axs[i].title.set_text(c_test_genomes[orig_idx]+", F1: "+str(round(f1_best,2)))
        axs[i].set_yticklabels(['','Reconstructed', 'Corrupted', 'Original'])
        
        if i == 2:
            axs[i].set_xlabel('Percent (%)')
        
        print(c_test_genomes[orig_idx],"F1: "+str(f1_best))
        print("generated genome:",data1)
        print("generated genome:",perc1)
        print()
        
    plt.tight_layout()

    return fig

def snowplot(f1s, c_test_genomes, tns, fps, fns, tps, uncorrupted, corrupted, idx):
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


def genome_heatmap2(corrupted_test, idx, model, f1s, tns, fps, fns, tps, binary_pred):
    from matplotlib.colors import LinearSegmentedColormap
    import sklearn as sk
    # TN = black
    # TP = green
    # FN =  magenta
    # FP = yellow
    # padding = white
    colours = ['black', 'green', 'magenta', 'yellow', 'white']
    cmap_name = 'my_list'
    
    n_features = int(corrupted_test.shape[1]/2)
    # set up dimensions of pixel rectangle
    n_extension = 100*99 - n_features
    n_rows = 99
    n_cols = 100
    
    # Get corrupted version of genome
    corrupted = corrupted_test[idx][:n_features].tolist()
    corrupted.extend([4] * n_extension) # 100*100 - n_features
    corrupted = np.reshape(corrupted, (n_rows, n_cols))
    cm = LinearSegmentedColormap.from_list(cmap_name, colours, N=len(colours))
    # Get uncorrupted version of genome
    uncorrupted = corrupted_test[idx][n_features:].tolist()
    uncorrupted.extend([4] * n_extension) # 100*100 - n_features
    uncorrupted = np.reshape(uncorrupted, (n_rows, n_cols))  
    
    # Get predicted uncorrupted version of genome
    corr_genome = corrupted_test[idx][:n_features]
    true_genome = corrupted_test[idx][n_features:]
#    model.eval()
#    pred = model.forward(corr_genome)
#    pred = pred[0].tolist()
#    binary_pred = [1 if i > 0.5 else 0 for i in pred]
    
    binary_pred = binary_pred[idx]

    #print("Num on bits",int(sum(corr_genome)))
#    print("Original num on bits",int(sum(true_genome)))
#    print("Pred num on bits",int(sum(binary_pred)))
    tn = tns[idx]
    fp = fps[idx]
    fn = fns[idx]
    tp = tps[idx]
    #tn, fp, fn, tp = sk.metrics.confusion_matrix(true_genome, binary_pred).flatten()
    print("tn",tn, "fp",fp, "fn",fn, "tp",tp)
    #print("F1", sk.metrics.f1_score(true_genome, binary_pred, zero_division=0))
    print(f1s[idx])
    
    
    colour_pred = []
    for i in zip(binary_pred, corr_genome, true_genome):
        if i[0] == i[2] == 1: # TP
            colour_pred.append(1) 
        elif i[0] == i[2] == 0: # TN
            colour_pred.append(0) 
        elif i[0] == 0 and i[2] == 1: # False negative
            colour_pred.append(2)
        else: # False positive
            colour_pred.append(3)
            
    print(len(colour_pred))
    print("n_features",n_features,"len(colour_pred)",len(colour_pred),"n_extension",n_extension)
    colour_pred.extend([4] * n_extension) # 100*100 - n_features
    print(len(colour_pred))
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

def generate_out2(generated, gen_idx, all_kos, BASE_DIR, save_to):
	metab_recon = generated[gen_idx]
	
	keeps= []
	for i in range(len(metab_recon)):
	    if metab_recon[i] == 1: keeps.append(i)
	
	ko_ids = []
	for idx in keeps:
	    ko_ids.append(all_kos[idx])
	    
	with open(BASE_DIR+'seq_dict.pkl', 'rb') as handle:
	    seq_dict = pickle.load(handle)
	
	with open(save_to, 'w') as handle:
	    for prot in ko_ids:
	        handle.write(">"+prot+"\n")
	        handle.write(seq_dict[prot]+"\n")   

def generated_out(df, generated_idx, all_kos, save_to, BASE_DIR):
    with open(BASE_DIR+'seq_dict.pkl', 'rb') as handle:
        seq_dict = pickle.load(handle)
        
    metab_recon = torch.Tensor(df.iloc[generated_idx]).reshape(1,-1)

    # get indices that are turned on in the prediction
    on_idx = [i[1] for i in (metab_recon == 1).nonzero().tolist()]

    ko_ids = []
    for idx in on_idx:
        ko_ids.append(all_kos[idx])

    with open(save_to, 'w') as handle:
        for prot in ko_ids:
            handle.write(">"+prot+"\n")
            handle.write(seq_dict[prot]+"\n")

def arch_root(all_kos):
	# Create archaea outgroup
	path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/annotations/'
	file = open(path+'barc_annotations.txt').readlines()
	file = list(map(str.strip, file))
	
	barc_kos = []
	for s in file:
	    if "<a href=" in s:
	        x = s.split()[2]
	#             if "K00668" in s:
	#                 print("s",s)
	#                 print("x", x)
	#                 print()
	        if re.match(r'[K]\d{5}', x): 
	            barc_kos.append(x) #[K]\d{5}
	
	barc_vec = []
	for ko in all_kos:
	    if ko in barc_kos:
	        barc_vec.append(1)
	    else:
	        barc_vec.append(0)
	
	return barc_vec

def df_genus_genomes(idx_best, tla_best, tla_to_tnum, tnum_to_tax, binary_pred, test_genomes, test_data, train_genomes, train_data):
	"""
	Given a generated genome vector index, returns a df where rows are genomes and columns are genes, for:
		- The generated genome vector
		- The test set genome vector from which the input mods were sampled
		- All genome vectors of the same genus in the test set
	"""

	tnum_best = tla_to_tnum[tla_best]
	genus_best = tnum_to_tax[tnum_best][5]
	
	# Get reconstruction genome vector (bruc_reconstruction)
	df_generated = pd.DataFrame(binary_pred[idx_best,:], columns=['Generated']).T

	# get row for uncorrupted B. abortus 
	bruc_best_idx = test_genomes.index(tnum_best) # indexing is different for binary_pred and test_genomes
	df_orig = pd.DataFrame(test_data[bruc_best_idx,:], columns=[tnum_to_tax[tnum_best][6]]).T

	# Create df with all Brucella genomes of interest
	df = pd.concat([df_generated, df_orig])


	# add data from training set Brucella genomes to df
	labels = ['Generated', tnum_to_tax[tnum_best][6]]
	for tnum in tnum_to_tax:
	    genus = tnum_to_tax[tnum][5]
	    if (genus == genus_best) and (tnum in train_genomes):
	        i = train_genomes.index(tnum)
	        train_genome = pd.DataFrame(train_data[i,:]).T
	        df = df.append(train_genome, ignore_index = True)
	        labels.append(tnum_to_tax[tnum][6])

	df.index = labels

	return df

def diff_heatmap(df):
	labels = df. index
	
	# Calculate num substitutions between genomes
	data = []
	for i, row1 in df.iterrows():
	    diffs = []
	    for j, row2 in df.iterrows():
	        row1 = np.array(row1)
	        row2 = np.array(row2)
	        simm = np.matmul(row1, row2)
	        diff = (np.sum(row1) - simm) + (np.sum(row2) - simm)
	        diffs.append(diff)
	    data.append(diffs)
	
	diff_df = pd.DataFrame(data, labels, columns=labels)
	
	# plot
	fig = sns.clustermap(diff_df, cmap='Blues', annot=True, vmin=0, vmax=2000)

	return fig

def map_proc_mod():
    mod_names = {}
    process_to_mod = {}
    path = "/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/kegg_modules.txt"
    file = open(path).readlines()
    file = list(map(str.strip, file))
    
    type_proc = ""
    for s in file:
        if s[0] == "D": 
            mod_names[s.split()[1]] = ' '.join(s.split()[2:]).split('[')[0].split(',')[0]
    return mod_names 
    
def compare_inputs(test_input_mods, idx_best, org_to_mod_to_kos, train_genomes, tla_to_tnum, mod_names):
	# What input modules were used for generated brucella abortus genome?
	gen_mods = test_input_mods[idx_best]	
	
	# Which orgs have those mods?
	mod_count = defaultdict(int)
	all_ten = []
	for tla in org_to_mod_to_kos:
	    try:
	        tnum = tla_to_tnum[tla]
	    except KeyError: pass
	    if tnum not in train_genomes: continue
	    mods = list(org_to_mod_to_kos[tla].keys())
	    for mod in mods:
	        if mod in gen_mods:
	            mod_count[mod] += 1 
	    
	    # of input mods to brucella genome, how many genomes have all ten?
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
	
def compare_venn(genome1, genome2, genome3, df):
	from matplotlib_venn import venn3, venn3_circles
	
	canis = df.loc[genome1].tolist()
	microti = df.loc[genome2].tolist()
	generated = df.loc[genome3].tolist()
	
	canis2 = []
	microti2 = []
	generated2 = []
	for i in range(len(canis)):
	    if canis[i] == microti[i] == generated[i] == 1:
	        canis2.append(str(i)+'shared3')
	        microti2.append(str(i)+'shared3')
	        generated2.append(str(i)+'shared3')
	    elif canis[i] == microti[i] == 1:
	        canis2.append(str(i)+'sharedCM')
	        microti2.append(str(i)+'sharedCM')  
	    elif canis[i] == generated[i] == 1:
	        canis2.append(str(i)+'sharedCG')
	        generated2.append(str(i)+'sharedCG')   
	    elif microti[i] == generated[i] == 1:
	        microti2.append(str(i)+'sharedCG')   
	        generated2.append(str(i)+'sharedCG')
	    elif canis[i] == 1:
	        canis2.append(str(i)+'unique')
	    elif microti[i] == 1:
	        microti2.append(str(i)+'unique')
	    elif generated[i] == 1:
	        generated2.append(str(i)+'unique')
	
	fig, ax = plt.subplots(figsize=(6, 6))
	fig = venn3([set(canis2), set(microti2), set(generated2)], 
set_labels = (genome1, genome2, genome3))
	
	return fig, ax

def get_ten_closest(index_now, tnum_now, test_genomes, train_genomes, test_data, train_data, binary_pred, tnum_to_tax):

	# Get reconstruction genome vector (bruc_reconstruction)
	generated = binary_pred[index_now,:]
	# Get GV from which it was derived
	bruc_best_idx = test_genomes.index(tnum_now) # indexing is different for binary_pred and test_genomes
	orig = test_data[bruc_best_idx,:]
	
	
	hammings = []
	idxs = []
	for i, row in enumerate(train_data):
	    hl = hamming_loss(generated, row)
	    hammings.append(hl)
	    idxs.append(i)
	    
	hammings, train_genomes2, idxs = zip(*sorted(zip(hammings, train_genomes, idxs), reverse=False))
	hamm_10 = hammings[:10]
	closest_genomes = train_genomes2[:10]
	idx_10 = idxs[:10]

	labels = [tnum_to_tax[i][6] for i in closest_genomes] + [tnum_to_tax[tnum_now][6]+'*', 'Generated']
	
	ten_df = pd.DataFrame(np.vstack((train_data[idx_10, :], test_data[bruc_best_idx,:], generated)), labels)

	return ten_df, closest_genomes
	
def genus_boxplot_partial(c_test_genomes, tla_to_tnum, tnum_to_tax, tax_groups, f1s):
	import pylab as P
	
	genus_count = defaultdict(int)
	genus_f1 = defaultdict(list)
	for i, tla in enumerate(c_test_genomes):
	    tnum = tla_to_tnum[tla]
	    genus = tnum_to_tax[tnum][5]
	    genus_count[genus] = tax_groups['genus'].count(genus)
	    genus_f1[genus].append(f1s[i])
	
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
	
	return [group_0, group_1, group_2, group_3, group_4, group_5]

def genus_boxplot(c_test_genomes, tla_to_tnum, tnum_to_tax, tax_groups, f1s):
	import pylab as P
	
	genus_count = defaultdict(int)
	genus_f1 = defaultdict(list)
	for i, tla in enumerate(c_test_genomes):
	    tnum = tla_to_tnum[tla]
	    genus = tnum_to_tax[tnum][5]
	    genus_count[genus] = tax_groups['genus'].count(genus)
	    genus_f1[genus].append(f1s[i])
	
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
	
	fig, ax = plt.subplots(figsize=(6, 6))
	#ax.boxplot([group_0, group_1, group_2, group_3, group_4, group_5], labels=[0,1,2,3,4,5], sym='.')
	# ax.set_title('', fontsize=10)
	# ax.set_xlabel('Number of same-genus genomes in the training set')
	# ax.set_ylabel('F1 score')
	
	for i, group in enumerate([group_0, group_1, group_2, group_3, group_4, group_5]):
	    x = np.random.normal(1+i, 0.08, size=len(group)) # scatter
	    P.plot(x, group, color='#4fc657', marker='.', linestyle="None", alpha=0.5, markersize = 10) 
	
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
	
	return fig, [group_0, group_1, group_2, group_3, group_4, group_5]
	
def genus_boxplot_stats(groups):
	from pingouin import anova
	from statsmodels.stats.multicomp import pairwise_tukeyhsd
	
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
