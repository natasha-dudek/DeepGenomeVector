import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import math
from collections import defaultdict
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
import random

def learning_curve(train_y, test_y, type_curve, cirriculum):
    """
    Plots a learning curve
    
    Arguments:
    train_y -- list of values for training dataset (e.g.: loss scores, F1 score)
    test_y -- list of values for test dataset (e.g.: loss scores, F1 score)
    type_curve -- [optimization | performance | any_string] used to put descriptive title + axis labels on plot
    
    Returns:
    nothing -- plots figure
    """
    if len(train_y) != len(test_y):        
        print("The number of training losses is not the same as the number of test losses")
        
        diff_len = len(train_y) - len(test_y)
        
        longer = "train"
        if diff_len < 0:
            longer = "test"
        
        print(("Dropping %s instances from the %sing set to make lengths equal") % (abs(diff_len), longer))
    
    x_losses = [*range(len(train_y))]
    
    if len(train_y) != len(test_y):
        if longer == "train":
            x_losses = x_losses[:len(test_y)]
            train_y = train_y[:len(test_y)]
        else:
            x_losses = x_losses[:len(train_y)]
            test_y = train_y[:len(train_y)]
            
    fig=plt.figure(figsize=(20, 7))
    ax=fig.add_subplot()
    ax.plot(x_losses,train_y,marker='o', c='b',label='Training',fillstyle='none')
    ax.plot(x_losses,test_y,marker='o',c='g',label='CV',fillstyle='none')
    
    plt.legend(loc=1)
    
    
    if type_curve == "optimization":
        plt.title("Optimization Learning Curve")
        plt.ylabel("Loss")
        plt.xlabel("Experience")
        plt.yscale('log')
    elif type_curve == "performance":
        plt.title("Performance Learning Curve")
        plt.ylabel("F1 score")
        plt.xlabel("Experience")
    else:
        plt.title(type_curve+" Learning Curve")
        plt.xlabel("Experience (batches)")
    
    if cirriculum:
        switch =  int(len(x_losses) / 3)
        x1 = switch - 1
        x2 = switch
        x3 = 2*switch -1
        x4 = 2*switch 
        ax.axvspan(x1, x2, alpha=0.5, color='black')
        ax.axvspan(x3, x4, alpha=0.5, color='black')
        
    return fig
    
    
def calc_svd(data):
    """
    Perform truncated SVD on data (note: SVD works efficiently on sparse matrices, unlike PCA)
    
    Arguments:
    data -- torch tensor
    n_components -- desired dimensionality of output data
    
    Returns:
    svd_result -- reduced version of input data in the form of a numpy array
    """
    
    # First determine how many components to use (upper bound of 50 = max for input to tSNE)
    # I.e. as many as it takes to capture 99% of the variance (or 50, whichever is lower)
    n_components = 50
    svd = TruncatedSVD(n_components)
    svd.fit_transform(data)
    var_explained = svd.explained_variance_ratio_.cumsum()
    for i, var in enumerate(var_explained):
        if var >= 0.99:
            n_components = i + 1
            break

    svd2 = TruncatedSVD(n_components)    
    svd_result = svd2.fit_transform(data)
    
    return svd_result, n_components

def get_tax(test_tax_dict, genome_to_tax):
    
    # for every genome's index, get T number, use T number to get tax -> phylum
    #num_to_genome = {v: k for k, v in genome_to_num.items()}
    
    tax_labels = []
    
    for idx in range(len(test_tax_dict)):
        tax = genome_to_tax[test_tax_dict[idx]]
        phylum = tax.split(";")[1].split("p__")[1].strip("*")
        if phylum == "Proteobacteria":
            phylum = tax.split(";")[2].split("c__")[1].strip("*")
        tax_labels.append(phylum)

    cpr_list = ['Candidatus_Berkelbacteria', 'Candidatus_Campbellbacteria', 'Candidatus_Beckwithbacteria', 'Candidatus_Peregrinibacteria', 'Candidatus_Wolfebacteria', 'Candidatus_Saccharibacteria', 'Candidatus_Woesebacteria', 'candidate_division_WWE3']

    cpr_labels = []
    for phylum in tax_labels:
        out = ''
        if phylum in cpr_list:
            out = 'cpr'
        else:
            out = phylum
        cpr_labels.append(out)
    
    return cpr_labels
    
def plot_tSNE(embeddings, test_data, num_to_genome, genome_to_tax, test_tax_dict):
    """
    Generate and plot tSNE values for embeddings
    
    Steps:
    Reduces dimensionality of embeddings to 50 via truncated SVD (tSNE does not work well with high dimensional data)
    Uses reduced dimensionality embeddings to calculate tSNE values 
    Plots tSNE in 2D
    
    Arguments: 
    embedding -- embeddings, expecting torch tensor
    data -- expecting torch.utils.data.dataset.Subset created by pytorch's get_splits function
    
    Returns:
    nothing, generates a plot
    """
    
    # Tutorial: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    
    # First ensure embeddings are formatted as req'd for sklearn truncated SVD
    embeddings = embeddings.detach().numpy()
    embeddings = embeddings.astype(np.float)
    embeddings = torch.from_numpy(embeddings)
    
    
    # Reduce number of dimensions w/ truncated SVD (should not use >50 input features for tSNE)
    # Use truncated SVD instead of PCA bc sparse matrix
    #svd_result = calc_svd(embeddings, 50) # detach grad from embedding (don't need)
    svd_result, n_components = calc_svd(embeddings)
        
    # Estimate hyperparams for t-SNE
    # Tutorial: https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
    perplexity = np.power(len(embeddings), 0.5)
    print("Using",n_components,"components for truncated SVD and a perplexity of",perplexity,"for tSNE")
        
    # Calculate tSNE values
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=10000)
    if n_components >=50:
        tsne_results = tsne.fit_transform(svd_result) 
    else:
        tsne_results = tsne.fit_transform(embeddings)
    
    #get tax labels
    tax_labels = get_tax(test_tax_dict, genome_to_tax)
    
    
    
    # Plot tSNE
    df = pd.DataFrame({
    'tsne-2d-one': tsne_results[:,0],
    'tsne-2d-two': tsne_results[:,1],
    'tax': tax_labels
    }
    )
    
    #plt.figure(figsize=(16,10))
    
    avail_markers= ['x','o','v','^','<']
    label_markers= avail_markers*math.ceil(len(set(tax_labels))/len(avail_markers))
    label_markers = label_markers[:len(set(tax_labels))]
    
    sns.lmplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # palette=sns.color_palette("hls", len(set(tax_labels))),
        fit_reg=False,
        markers=label_markers,
        data=df,
        legend="full",
        hue="tax",
        scatter_kws={"s": 10},
        height=8.27, 
        aspect=11.7/8.27)

def taxa_per_cluster(df):
    # Create bar plot showing the number of taxa with each module/KO
    mod_counts = list(df.sum(axis=0) )

    mod_counts.sort(reverse=True)
    
    num_to_plot = 1000
    plt.bar(list(range(len(mod_counts[:num_to_plot]))),mod_counts[:num_to_plot])
    plt.xticks([])
    plt.xlabel('Cluster')
    plt.ylabel('# taxa')
    plt.title("# taxa per cluster")
    plt.savefig('taxa_per_cluster.png')

def module_stats(df):
    mod_counts = list(df.sum(axis=0) )
    mod_counts.sort(reverse=True)
    
    genomes_per_clust = {}
    mod_names = list(df.columns)
    for mod, count in zip(mod_names, mod_counts):
        genomes_per_clust[mod] = count
    
    n_genomes = df.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    #fig.suptitle('Horizontally stacked subplots')
    
    num_to_plot = 1000
    
    ax1.bar(list(range(len(mod_counts[:num_to_plot]))),mod_counts[:num_to_plot])
    ax1.set_xticks([])
    ax1.set_xlabel('Module')
    ax1.set_ylabel('# taxa')
    ax1.set_title("# taxa per module")
    
    ax2.hist(genomes_per_clust.values(), bins=n_genomes)
    ax2.semilogy()
    ax2.set_xlabel('# taxa')
    ax2.set_ylabel('# module')
    ax2.set_title("# module shared by n taxa")
    ax2.set_xlim(1,n_genomes)
    ax2.axvline(x=(n_genomes - 1), color='red') # core = genes contained in 100% of the genomes
    num_softcore = int(math.ceil(0.95*n_genomes))
    ax2.axvline(x=(num_softcore-1), color='red') # softcore = genes contained in 95% of the genomes
    num_50p = int(math.ceil(0.5*n_genomes))
    ax2.axvline(x=(num_50p-1), color='red') # 50% (sort of cloud)
    
    plt.savefig('module_stats.png')
    plt.show()
    
def tax_distrib(df, genome_to_tax):

    tax = [genome_to_tax[i] for i in list(df.index)] # list(df.index) is genome names T01278
    
    domain_count = defaultdict(lambda: 0)
    for i in tax:
        domain = i.split(";")[0][3:].strip("*")
        if domain == "TM6": domain = "Bacteria"
        domain_count[domain] += 1
        
    phylum_count = defaultdict(lambda: 0)
    for i in tax:
        if "k__Bacteria" not in i: continue
        
        phylum = i.split(";")[1][3:].strip("*")
        if phylum == "Proteobacteria":
            phylum = i.split(";")[2][3:].strip("*")
        phylum_count[phylum] += 1
        
    phy_counts = []
    phy_names = []
    for key, value in sorted(phylum_count.items(), key=lambda x: x[1], reverse=False):
        key = key.strip("*")
        key = key[0].upper()+key[1:]
        key = key.replace("_", " ")
        if "division" in key:
            x = key.split("division")
            key = x[0]+"Division"+x[1]
        
        phy_counts.append(value)
        phy_names.append(key)
    
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))        
    
    ##### Plot domain-level composition
    
    wedges, texts = ax1.pie(domain_count.values(), wedgeprops=dict(width=0.5), startangle=-40)
    labels = [i+" (n="+str(domain_count[i])+")" for i in domain_count]
        
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax1.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    
    ax1.set_title("Domain-level composition of genomes in the dataset")
    
    ##### Plot bacterial phylum-level composition
    
#    print(phy_counts)
#    print(phy_names)
    
    ax2.barh(phy_names,phy_counts, align='center')
    #ax2.set_xticks([])
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Phylum')
    ax2.set_title("Phylum-level composition of bacterial genomes in the dataset")
    ax2.semilogx()
    
    fig.tight_layout()
    plt.show()
        
def genes_per_genome(df):
    num_genes = df.sum(axis=1).tolist()
    num_genes.sort(reverse=True)
    
    fig, ax = plt.subplots(figsize=(20, 7))
    _ = plt.hist(num_genes, bins=100)
    plt.title('Number of KOs/modules per genome')
    plt.xlabel('KOs/modules per genome')
    plt.ylabel('Frequency')

def hist_prob_ko(train_data):
    
    n_genomes = train_data.shape[0]
    
    train_data = train_data
    ko_prob = np.sum(train_data, axis=0) / n_genomes
    #ko_prob = np.sum(train_data.detach().numpy(), axis=0) / n_genomes
    

    
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.semilogy()
    _ = plt.hist(ko_prob, bins=100)
    plt.title('Probability distribution of any given gene being present in a genome')
    plt.xlabel('Fraction of genomes containing any given KO')
    plt.ylabel('Frequency')
        
    
def my_roc_curve(target, y_probas):
    """
    Performs ROC / AUC calculations and plots ROC curve
    
    Arguments:
    target (numpy.ndarray) -- uncorrupted version of genomes (n_genomes, n_features)
    y_probas (numpy.ndarray) -- prediction for corrupted genomes (n_genomes, n_features)
    """
    
    # ROC/AUC calculations
    fpr = dict()
    tpr = dict()
    roc_auc = dict()    
    for i in range(target.shape[1]):
        _, n = target.shape
        print("num 1s",np.sum(target[:, i]))
        print("num 0s",n - np.sum(target[:, i]))
        print("sum 1s and 0s", n)
        fpr[i], tpr[i],thresh = roc_curve(target[:, i], y_probas[:, i])
        print("target[:, i]", np.isnan(np.max(target[:, i])))
        print("y_probas[:, i]", np.isnan(np.max(y_probas[:, i])))
#        print("thresh",thresh)
        #print(fpr[i], tpr[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    n_examples = 50 # will plot 50 example genes on ROC curve
    
    # get colours for plotting
    cm = plt.cm.get_cmap('gist_rainbow')
    c = np.linspace(0, 1, 50) # start, stop, how_many
    colours = [cm(i) for i in c]
    colours = colours*2
    
    # plot
    fig, ax = plt.subplots(figsize=(20, 7))
    a = random.sample(range(target.shape[1]), 50)
    for i in range(len(a)):
        plt.plot(fpr[a[i]], tpr[a[i]], color=colours[i], alpha=0.5,
             lw=2) #, label=cluster_names[i]+" (AUC = %0.2f)" % roc_auc[i])
    plt.plot(fpr["micro"], tpr["micro"], color='red', 
             lw=5, label='Micro-average (AUC = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for micro-average + 50 randomly selected genes')
    plt.legend(loc="lower right")
    #plt.show()
    
    return fig