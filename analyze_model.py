from collections import defaultdict
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn as sk
import torch

from genome_embeddings import data_viz
from genome_embeddings import train_test

def load_data(path, test_data):
    """
    Loads y_probas prediction from a model
    
    Arguments:
    path (str) -- path to y_probas.pt file
    test_data (tensor) -- all genomes in test set, corrupted + uncorrupted concatenated together
    
    Returns:
    y_probas (tensor) -- prediction probabilities
    target (numpy.ndarray) -- uncorrupted versions of each genome in test set
    num_features (int) -- # of features in test_data (test_data.shape[1] / 2)
    """
    y_probas = torch.load(path)
    # first convert test_data from subset -> tensor, split corrupt vs target sets
    tensor_test_data = torch.tensor([i.numpy() for i in test_data]).float()
    num_features = int(tensor_test_data.shape[1] / 2)
    corrupt_test_data = tensor_test_data[:,:num_features]
    target = tensor_test_data[:,num_features:].detach().numpy()
    
    return y_probas, target, num_features

def roc_curve(y_probas, target, save_path):
    """
    Uses data_viz.my_roc_curve to plot and save ROC curve
    
    Arguments:
    y_probas (tensor) -- prediction probabilities
    target (numpy.ndarray) -- ground truths
    save_path (str) -- path to dir where you want to save image (must end with "/")
    
    Returns:
    None
    """
    
    roc = data_viz.my_roc_curve(target, y_probas.numpy())
    roc.savefig(save_path+"roc.png", dpi=200, bbox_inches='tight')
    
def confusion_mat(y_probas, target, replacement_threshold):
    """
    Calculates confusion matrix metrics (TP, TN, FP, FN) plus # on bits per genome plus F1 scores
    
    Arguments:
    y_probas (tensor) -- prediction probabilities
    target (numpy.ndarray) -- ground truths
    replacement_threshold (float) -- threshold at which to convert prediction probability to 1 vs 0
    
    Returns:
    cms (list of lists) -- confusion matrix metrics
        cms[0] (list of ints) -- # of TNs 
        cms[1] (list of ints) -- # of FPs 
        cms[2] (list of ints) -- # of FNs 
        cms[3] (list of ints) -- # of TPs 
        cms[4] (list of ints) -- # of on bits per genome
    f1s (list of floats) -- F1 scores 
    b_pred -- binarized predictions
    """

    b_pred = train_test.binarize(y_probas, replacement_threshold)
    
    f1s = []
    cms = []
    for i in range(0,len(b_pred)):
        f1 = sk.metrics.f1_score(target[i], b_pred[i])
        cm = sk.metrics.confusion_matrix(target[i], b_pred[i]).flatten().tolist() #tn, fp, fn, tp
        cm.append(target[i].tolist().count(1)) # n_genes that should be on in each genome 
        f1s.append(f1)
        cms.append(cm)
    print(max(f1s), min(f1s), np.median(f1s))
    return cms, f1s, b_pred

def cm_histogram(tp, tn, fp, fn, n_genes, n_features, DATA_FP):
    """
    Plot distribution of # TPs, FPs, FNs, TNs across genomes
    
    Note this is of limited use as the number of on genes per genome varies widely and there is not necessarily a good method for standardizing
    
    tp (list of ints) -- list of # true positives per genome
    tn (list of ints) -- list of # true negatives per genome
    fp (list of ints) -- list of # false positives per genome
    fn (list of ints) -- list of # false negatives per genome
    n_genes (list of ints) -- list of # on bits per genome
    n_features (int) -- number of features
    DATA_FP -- dir to save data (must end with a "/")
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,7))
    
    ax1.hist(tp)
    ax2.hist(fp)
    ax4.hist(tn)
    ax3.hist(fn)

    ax1.set_title("TP")
    ax2.set_title("FP")
    ax4.set_title("TN")
    ax3.set_title("FN")
    
    ax1.set_xlabel('# TP per genome')
    ax2.set_xlabel('# FP per genome')
    ax4.set_xlabel('# TN per genome')
    ax3.set_xlabel('# FN per genome')
    
    ax1.legend("Median:"+str(np.median(tp)))
    
    ax1.set_ylabel('Frequency')
    ax3.set_ylabel('Frequency')
    fig.suptitle("Histogram of TPs, TNs, FPs, and FNs on test set")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.savefig(DATA_FP+"hist_confusion_matric.png", dpi=200, bbox_inches='tight')
    
    print("TP",np.median(tp), ", FP",np.median(fp), ", FN",np.median(fn), ", TN",np.median(tn))
    
def tp_fp_fn_tn(cms):
    """
    Unpack TP, FP, FN, TN from list cm
    
    Arguments:
    cms (list of lists) -- confusion matrix metrics
        cms[0] (list of ints) -- # of TNs 
        cms[1] (list of ints) -- # of FPs 
        cms[2] (list of ints) -- # of FNs 
        cms[3] (list of ints) -- # of TPs 
        cms[4] (list of ints) -- # of on bits per genome
    
    Returns:
    tn (list of ints) -- # of TNs 
    fp (list of ints) -- # of FPs 
    fn (list of ints) -- # of FNs 
    tp (list of ints) -- # of TPs 
    n_genes (list of ints) -- # of on bits per genome    
    
    """
    tn = [i[0] for i in cms]
    fp = [i[1] for i in cms]
    fn = [i[2] for i in cms]
    tp = [i[3] for i in cms]
    n_genes = [i[4] for i in cms]
    
    return tn, fp, fn, tp, n_genes

def phylum_stats(genome_to_tax, test_tax_dict, genome_idx_test, cms):
    """
    Creates a dict with useful stats per bacterial phylum (avg F1, TP, FP, FN, TN, n_genes)
    
    Arguments:
    genome_to_tax (dict) -- maps genome ID # to taxonomic ID
    test_tax_dict (dict) -- maps row of filtered test set (bact only) to an original genome ID #
    genome_idx_test (dict) -- maps row of corrupted genome to row of uncorrupted genome in filtered test set
    cms (list of lists) -- confusion matrix metrics
        cms[0] (list of ints) -- # of TNs 
        cms[1] (list of ints) -- # of FPs 
        cms[2] (list of ints) -- # of FNs 
        cms[3] (list of ints) -- # of TPs 
        cms[4] (list of ints) -- # of on bits per genome    

    Returns:
    phylum_stats (dict) -- useful stats for genomes in each phylum
        keys -- phylum
        values -- lists of F1, TN, FP, FN, TP, n_genes for all genomes from phylum
    """
    
    # unpack cms
    tn, fp, fn, tp, n_genes = tp_fp_fn_tn(cms)
    
    # figure out taxonomy of each genome
    genome_ids = [genome_to_tax[test_tax_dict[genome_idx_test[i]]] for i in range(len(f1s))]
    phylum_stats = defaultdict(lambda: defaultdict(list))
    
    for i, tax in enumerate(genome_ids):
        phylum = tax.split("p__")[1].split(";")[0]
        phylum_stats[phylum]["f1"].append(f1s[i])
        phylum_stats[phylum]["tn"].append(tn[i])
        phylum_stats[phylum]["fp"].append(fp[i])
        phylum_stats[phylum]["fn"].append(fn[i])
        phylum_stats[phylum]["tp"].append(tp[i])
        phylum_stats[phylum]["n_genes"].append(n_genes[i])

    return phylum_stats

def tax_cm_histogram_PR(phylum_stats, save_path):
    """
    Plot median precision and recall per bacterial phylum
    
    Arguments:
    phylum_stats (dict) -- useful stats for genomes in each phylum
        keys -- phylum
        values -- lists of F1, TN, FP, FN, TP, n_genes for all genomes from phylum
    save_path (str) -- directory to save figure (must end with "/")
    
    Returns:
    None
    """
    
    phyla_to_num = {}
    num_to_phyla = {}
    for idx, tax in enumerate(phylum_stats):
        num_to_phyla[idx] = tax.strip("*")
        phyla_to_num[tax.strip("*")] = idx
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,7))
    
    precision = {}
    recall = {}
    precision_mad = {}
    recall_mad = {}
    # iterate through each phylum i
    for i in phylum_stats:

        prs = [(tp/(tp+fp)) for tp,fp in zip(phylum_stats[i]["tp"], phylum_stats[i]["fp"])]
        res = [(tp/(tp+fn)) for tp,fn in zip(phylum_stats[i]["tp"], phylum_stats[i]["fn"])]
        precision_mad[i.strip("*")] = scipy.stats.median_absolute_deviation(prs)
        recall_mad[i.strip("*")] = scipy.stats.median_absolute_deviation(res)
        
        precision[i.strip("*")] = np.median(prs)
        recall[i.strip("*")] = np.median(res)

        
    sorted_precision = sorted(precision.items(), key=lambda x: x[1], reverse=True)
    sorted_recall = sorted(recall.items(), key=lambda x: x[1], reverse=True)
    sorted_pmad = [precision_mad[phylum[0]] for phylum in sorted_precision]
    sorted_rmad = [recall_mad[phylum[0]] for phylum in sorted_recall]
    
    ax1.grid(True, linestyle="--", color='grey', alpha=0.5)
    ax2.grid(True, linestyle="--", color='grey', alpha=0.5)
    
    ax1.bar(*zip(*sorted_precision), yerr=sorted_pmad)
    p_labels = [phyla_to_num[i[0]] for i in sorted_precision]
    ax2.bar(*zip(*sorted_recall), yerr=sorted_rmad)
    r_labels = [phyla_to_num[i[0]] for i in sorted_recall]

    ax1.set_title("Median precision per phylum")
    ax2.set_title("Median recall per phylum")
    
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Count')
    ax1.set_xlabel('Phylum')
    ax2.set_xlabel('Phylum')
    
    ax1.set_xticklabels(p_labels, rotation=0)
    ax2.set_xticklabels(r_labels, rotation=0)
    
    fig.tight_layout()
    
    textstr = "Phylum \n"
    for i in num_to_phyla:
        textstr += str(i)+": "+num_to_phyla[i]+"\n"
    plt.text(30, 0,textstr)#, horizontalalignment='center', verticalalignment='center')
    
    plt.savefig(save_path+"hist_confusion_matric_tax_PR.png", dpi=400, bbox_inches='tight')
    
def tax_cm_histogram(phylum_stats, save_path):
    """
    
    Plots # TP, FP, FN, TN per bacterial phylum
    
    Note: different genomes within a single phylum will have different gene counts so this isn't necessarily super meaningul -- one day may want to standardize by # of on bits per genome
    
    Arguments:
    phylum_stats (dict) -- useful stats for genomes in each phylum
        keys -- phylum
        values -- lists of F1, TN, FP, FN, TP, n_genes for all genomes from phylum
    save_path (str) -- directory to save figure (must end with "/")
    
    Returns:
    None
    """

    phyla_to_num = {}
    num_to_phyla = {}
    for idx, tax in enumerate(phylum_stats):
        num_to_phyla[idx] = tax.strip("*")
        phyla_to_num[tax.strip("*")] = idx
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,7))
    
    tax_tp = {}
    tax_fp = {}
    tax_fn = {}
    tax_tn = {}
    for i in phylum_stats:
        tax_tp[i.strip("*")] = np.median(phylum_stats[i]["tp"])
        tax_fp[i.strip("*")] = np.median(phylum_stats[i]["fp"])
        tax_fn[i.strip("*")] = np.median(phylum_stats[i]["fn"])
        tax_tn[i.strip("*")] = np.median(phylum_stats[i]["tn"])
    
    sorted_tp = sorted(tax_tp.items(), key=lambda x: x[1], reverse=True)
    sorted_fp = sorted(tax_fp.items(), key=lambda x: x[1], reverse=True)
    sorted_fn = sorted(tax_fn.items(), key=lambda x: x[1], reverse=True)
    sorted_tn = sorted(tax_tn.items(), key=lambda x: x[1], reverse=True)

    ax1.bar(*zip(*sorted_tp))
    #tp_labels = [phyla_to_num[i[0]] for i in sorted_tp]
    tp_labels = [phyla_to_num[i[0]] for i in sorted_tp]
    ax2.bar(*zip(*sorted_fp))
    fp_labels = [phyla_to_num[i[0]] for i in sorted_fp]
    ax3.bar(*zip(*sorted_fn))
    fn_labels = [phyla_to_num[i[0]] for i in sorted_fn]
    ax4.bar(*zip(*sorted_tn))
    tn_labels = [phyla_to_num[i[0]] for i in sorted_tn]

    ax1.set_title("TP")
    ax2.set_title("FP")
    ax4.set_title("TN")
    ax3.set_title("FN")
    
    ax1.set_ylabel('Count')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Phylum')
    ax4.set_xlabel('Phylum')
    
    ax1.set_xticklabels(tp_labels, rotation=0)
    ax2.set_xticklabels(fp_labels, rotation=0)
    ax3.set_xticklabels(fn_labels, rotation=0)
    ax4.set_xticklabels(tn_labels, rotation=0)

    fig.tight_layout()
    
    textstr = "Phylum \n"
    for i in num_to_phyla:
        textstr += str(i)+": "+num_to_phyla[i]+"\n"
    plt.text(30, 0,textstr)   

    plt.savefig(save_path+"hist_confusion_matric_tax.png", dpi=200, bbox_inches='tight')

def print_f1_per_phylum(phylum_stats):
    """
    Prints median F1 score and median # genes per genome for each bacterial phylum
    
    Arguments:
    phylum_stats (dict) -- useful stats for genomes in each phylum
        keys -- phylum
        values -- lists of F1, TN, FP, FN, TP, n_genes for all genomes from phylum    
    """
    
    for i in phylum_stats:
        print(i.strip("*"), round(np.median(phylum_stats[i]["f1"]),2),
              np.median(phylum_stats[i]["n_genes"]))

def genes_f1(f1s, n_genes, save_path):
    """
    Plots # genes in genome vs F1 score for all genomes in test set (slow)
    
    Arguments:
    f1s (list of floats) -- list of F1 scores for all genomes
    n_genes (list of floats) -- list of # of on genes per genome for all genomes
    save_path (str) -- directory to save figure (must end with "/")
    
    Returns:
    None
    """
    from matplotlib import cm
    
    plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    xs = []
    ys = []

    for x, y in zip(n_genes, f1s):
        plt.scatter(x, y, color='b')
        xs.append(x)
        ys.append(y)
        
    # fit trendline
    coeffs = np.polyfit(xs, ys, 1)
    p = np.poly1d(coeffs)
    
    # calculate r2 value
    yhat = p(xs)
    ybar = np.sum(ys)/len(ys)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((ys - ybar)**2)
    r_sq = ssreg / sstot
    
    plt.plot(xs,p(xs),'r-')
    plt.text(1500,0.5,"y=%.4fx + %.2f, R2=%.2f"%(coeffs[0],coeffs[1], r_sq))
    plt.xlabel("# genes in genome")
    plt.ylabel("F1 score")
    plt.savefig(save_path+"f1_n_genes.png", dpi=400, bbox_inches='tight')

def load_brite(path):
    """
    Will scan document and pull out all KO numbers (e.g. after copy pasting brite paths from KEGG into txt doc)
    
    Arguments:
    path (str) -- path to txt file to open
    
    Returns:
    kos (list) -- list of unique KO numbers found in document
    """
    
    file = open(path, "r")
    file = list(map(str.strip, file))
    
    kos = []
    for i in file:
        kos.extend(re.findall("[K][0-9]{5}", i))
    return list(set(kos))

def pathway_perf(transporters_path, abx_res_path, rps_path):
    """
    Create dict with metabolic paths and the KEGG KO or Mod #s therein
    
    Arguments:
    transporters_path (str) -- path to file with transporters KOs
    abx_res_path (str) -- path to file with abx_res KOs
    rps_path (str) -- path to file with rp KOs
    
    Returns:
    metab (dict) -- keys: pathways, values: list of KOs or modules
    """
    
    transporters = load_brite(transporters_path)
    abx_res = load_brite(abx_res_path)
    rps = load_brite(rps_path)
    central_carbon = ["M00004", "M00006", "M00007", "M00580", "M00005", "M00008", "M00308", "M00633", "M00309"]
    # Energy acquisition
    carbon_fixation = ['M00165','M00166','M00167','M00168','M00169','M00172','M00171','M00170','M00173','M00376','M00375','M00374','M00377','M00579','M00620']
    methane_metab = ['M00567','M00357','M00356','M00563','M00358','M00608','M00174','M00346','M00345','M00344','M00378','M00422']
    nitrogen_metb = ['M00175','M00531','M00530','M00529','M00528','M00804']
    sulfur_metab = ['M00176','M00596','M00595']
    photosynthesis = ['M00161','M00163','M00597','M00598']
    atp_syn = ['M00144','M00145','M00142','M00143','M00146','M00147','M00149',
                'M00150','M00148','M00162','M00151','M00152','M00154','M00155',
                'M00153','M00417','M00416','M00156','M00157','M00158','M00159','M00160']
        
    metab = {"transporters":transporters, "abx_res":abx_res, "rps":rps, "central_carbon":central_carbon,
            "carbon_fixation":carbon_fixation, "methane_metab":methane_metab, "nitrogen_metb":nitrogen_metb,
            "sulfur_metab":sulfur_metab, "photosynthesis":photosynthesis, "atp_syn":atp_syn}
    
    return metab

def f1_per_brite(brite, cluster_names, b_pred_np, target):
    """
    Calculate the median F1 of genes within a BRITE category / pathway across all genomes
    
    Arguments:
    brite (list of str) -- list of KOs in BRITE group 
    cluster_names (str) -- name of KOs in test data (in same order as test data)
    b_pred_np (numpy.ndarray) -- binarized predictions (1's or 0's)
    target (tensor) -- uncorrupted versions of genomes
    
    Returns:
    np.median(f1s) -- median F1 value for genes in BRITE category
    """

    f1s = []
    for mod in brite:
        try:
            idx = cluster_names.index(mod)
            i_pred = b_pred_np[:,idx]
            i_target = target[:,idx]
            f1 = sk.metrics.f1_score(i_target, i_pred)
            f1s.append(f1)
            #print(mod, f1, list(tr[:,idx]).count(1))
        except ValueError: pass #print(mod,"not in cluster_names")
    return np.median(f1s)    

def f1_per_bright_group(metab, cluster_names, b_pred_np, target):
    """
    Print median F1 score of genes in a BRITE category
    
    Arguments: 
    metab (dict) -- keys: pathways, values: list of KOs or modules
    cluster_names (str) -- name of KOs in test data (in same order as test data)
    b_pred_np (numpy.ndarray) -- binarized predictions (1's or 0's)
    target (tensor) -- uncorrupted versions of genomes
    """
    print("brite_category, median_F1, n_genes_in_group")
    for i in metab:
        print(i, f1_per_brite(metab[i], cluster_names, b_pred_np, target), len(metab[i]))
        
        
        