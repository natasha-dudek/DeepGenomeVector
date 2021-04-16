import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import seaborn as sns
import torch
import math
from collections import defaultdict
from collections import OrderedDict
from scipy import interp
from sklearn.metrics import roc_curve, auc
import random

def learning_curve(train_losses, test_losses, train_f1s, test_f1s):
	"""
	Plots a learning curve
	
	Arguments:
	train_y -- list of values for training dataset (e.g.: loss scores, F1 score)
	test_y -- list of values for test dataset (e.g.: loss scores, F1 score)
	type_curve -- [optimization | performance | any_string] used to put descriptive title + axis labels on plot
	
	Returns:
	nothing -- plots figure
	"""
	
	plt.rcParams.update({'font.size': 16})
	
	x_losses = [*range(len(train_losses))]
	
	fig, axs = plt.subplots(2, figsize=(20, 10))
	
	axs[0].set_title("Optimization Learning Curve")
	axs[1].set_title("Performance Learning Curve")
	
	axs[0].set_ylim(10**4,10**7)
	axs[1].set_ylim(0,1)
	
	axs[0].plot(x_losses, train_losses, marker='.', c='#3385ff', label='Training', markersize=13)
	axs[0].plot(x_losses, test_losses, marker='.', c='#33cc33', label='CV', markersize=13)
	
	axs[1].plot(x_losses, train_f1s, marker='.', c='#3385ff', label='Training', markersize=13)
	axs[1].plot(x_losses, test_f1s, marker='.', c='#33cc33', label='CV', markersize=13)
	
	axs[0].set_xlim(0,x_losses[-1]+1)
	axs[1].set_xlim(0,x_losses[-1]+1)
	
	axs[0].set_ylabel('Loss (KLD + BCE)')
	axs[0].semilogy()
	axs[1].set_ylabel('F1 score')
	
	#axs[0].set_xlabel('Experience')
	axs[1].set_xlabel('Experience')
	
	axs[1].axhline(y=max(test_f1s), color='r', dashes=(1,1))
	print("max F1 score", max(test_f1s))
	
	axs[0].legend(loc="upper right")
	#axs[1].legend(loc="lower right")
	
	plt.tight_layout()
	
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
	embedding (tensor) -- embeddings
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
	
#	print(phy_counts)
#	print(phy_names)
	
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
	
	n_genomes = target.shape[0]
	n_genes = target.shape[1]
	
	# Calculate scores for each individual gene
	for i in range(n_genes):
		fpr[i], tpr[i], thresh = roc_curve(target[:, i], y_probas[:, i])
		
		if np.isnan(fpr[i]).any():
#			print("ERROR ON ROW i",i)
#			print("num 1s",np.sum(target[:, i]))
#			print("num 0s",n_genomes - np.sum(target[:, i]))
#			print("sum 1s and 0s", n_genomes)
#			print("target[:, i]", np.isnan(np.max(target[:, i])))
#			print("y_probas[:, i]", np.isnan(np.max(y_probas[:, i])))
			continue
		
			roc_auc[i] = auc(fpr[i], tpr[i])
	
	 # Calculate micro-average
#	fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_probas.ravel())
#	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	fpr_micro, tpr_micro, _ = roc_curve(target.ravel(), y_probas.ravel())
	roc_auc["micro"] = auc(fpr_micro, tpr_micro)

	
	# THIS IS INCLUDING MICROAVERAGE IN MACROAVERAGE CALC
	# Calculate macro-average
	# First aggregate all false positive rates
#	all_fpr = np.unique(np.concatenate([fpr[x] for x in range(target.shape[1]) if not np.isnan(fpr[x]).any()]))

	# Then interpolate all ROC curves at this points
#	mean_tpr = np.zeros_like(all_fpr)
#	for i in range(target.shape[1]):
#		if np.isnan(fpr[i]).any(): continue
#		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
#	mean_tpr /= target.shape[1]
#	roc_auc["macro"] = auc(all_fpr, mean_tpr)

	
	
	n_examples = 100 # will plot 50 example genes on ROC curve
	
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
	plt.plot(fpr_micro, tpr_micro, color='black', 
			 lw=5, label='Micro-average (AUC = %0.2f)' % roc_auc["micro"])
#	plt.plot(all_fpr, mean_tpr, color='blue', 
#			 lw=5, label='Macro-average (AUC = %0.2f)' % roc_auc["macro"])			 
	plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	ax.set_ylim([0,1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	#plt.title('ROC for 50 randomly selected genes + micro-average')
	plt.legend(loc="lower right")
	#plt.show()
	
	return fig

def genome_heatmap2(corrupted_test, idx, model, f1s, tns, fps, fns, tps, binary_pred):
	from matplotlib.colors import LinearSegmentedColormap
	import sklearn as sk
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
	print("Uncorrupted -- On:",str(int(sum(uncorrupted))),"Off:",str(int(n_features - sum(uncorrupted))))

	uncorrupted.extend([4] * n_extension) # 100*100 - n_features
	uncorrupted = np.reshape(uncorrupted, (n_rows, n_cols))  
	
	# Get predicted uncorrupted version of genome
	corr_genome = corrupted_test[idx][:n_features]
	print("Uncorrupted -- On:",str(int(sum(corr_genome))),"Off:",str(int(n_features - sum(corr_genome))))
	true_genome = corrupted_test[idx][n_features:]
#	model.eval()
#	pred = model.forward(corr_genome)
#	pred = pred[0].tolist()
#	binary_pred = [1 if i > 0.5 else 0 for i in pred]
	
	binary_pred = binary_pred[idx]

	#print("Num on bits",int(sum(corr_genome)))
#	print("Original num on bits",int(sum(true_genome)))
#	print("Pred num on bits",int(sum(binary_pred)))
	tn = tns[idx]
	fp = fps[idx]
	fn = fns[idx]
	tp = tps[idx]
	#tn, fp, fn, tp = sk.metrics.confusion_matrix(true_genome, binary_pred).flatten()
	print("Generated -- TN:",tn, "FP:",fp, "FN:",fn, "TP:",tp)
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

def kld_vs_bce(kld, bce):
	x = [i for i in range(len(kld))]
	kld = [int(i) for i in kld]
	bce = [int(i) for i in bce]
	plt.scatter(x,kld, c='b', marker='.', label='KLD')
	plt.scatter(x,bce, c='r', marker='.', label='BCE')
	plt.legend(loc='upper right')
	plt.xlabel("Experience")
	plt.ylabel("Loss")
	plt.yscale('log')
	#plt.savefig("/Users/natasha/Desktop/fig2.png")
	#return fig


def tax_distribution(c_train_genomes, c_test_genomes, mode):
	
	if mode == "CC":
		path = './'
		path2 = './'
	else:
		path = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/kegg_dataset/'
		path2 = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/'
	master_file = open(path+"downloaded_3LA.txt").readlines()
	master_file = map(str.strip, master_file)
	
	taxid_to_tla = {}
	
	for i in master_file:
		file = open(path+"kegg_dl/"+i).readlines()
		file = map(str.strip, file)
	
		threeLA = i.split("_")[0]
	
		for s in file:
			if "<b>Taxonomy</b></td><td>TAX:" in s:
				taxid = s.split("https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=")[1].split('">')[0]
	
		taxid_to_tla[taxid] = threeLA
	
	def final_sets(c_genomes, taxid_to_tla): # c_genomes is either c_train_genomes or c_test_genomes
		tax_path = open(path2+"ncbi_lineages_2020-05-04.csv").readlines()
		tax_path = map(str.strip, tax_path) 
	
		final_train_tlas = set(list(c_genomes))
		count = 0
		tax_dict = {}
		for i in tax_path:
			if count == 0:
				count += 1
				continue
			
			taxid = i.split(",")[0]
			try:
				tla = taxid_to_tla[taxid]
			except KeyError: continue
			
			if taxid in taxid_to_tla and tla in final_train_tlas:
				#print(taxid, taxid_to_tla[taxid])
				#print(i.split(",")[:8])
				tax_dict[tla] = i.split(",")[1:8]
		return tax_dict
	
	train_tax_dict = final_sets(c_train_genomes, taxid_to_tla)
	test_tax_dict = final_sets(c_test_genomes, taxid_to_tla)
	
	return train_tax_dict, test_tax_dict

def plot_tax_dist(c_train_genomes, c_test_genomes, mode):

	n_train = str(int(len(c_train_genomes)/100))
	n_test = str(int(len(c_test_genomes)/100))
	
	train_tax_dict, test_tax_dict = tax_distribution(c_train_genomes, c_test_genomes, mode)
	
	def phyla(tax_dict):
		dist = []
		for i in tax_dict:
			phylum = tax_dict[i][1]
			if phylum == "Proteobacteria":
				phylum = tax_dict[i][2]
			dist.append(phylum)
			
			
		dist2 = pd.DataFrame({'phylum': dist})
		dist2['freq'] = ''
		y = dist2.groupby('phylum').count() #.reset_index()
		
		return y

	y1 = phyla(train_tax_dict)
	y2 = phyla(test_tax_dict)
	
	#y3 = pd.merge(y1, y2, left_index=True, right_index=True)
	y3 = pd.merge(y1, y2, how='outer', left_index=True, right_index=True)
	y3['freq_y'] = y3['freq_y'].fillna(0)
	y3 = y3.sort_values('freq_x', ascending=False)
	
	#labels = y3.index.tolist()
	train_dist3 = y3['freq_x'].tolist()
	test_dist3 = y3['freq_y'].tolist()
	labels = [i for i in range(len(train_dist3))]

	x = np.arange(len(labels))  # the label locations
	width = 0.35  # the width of the bars
	
	fig, ax = plt.subplots()
	rects1 = ax.bar(x - width/2, train_dist3, width, label='Train (n='+n_train+')', color='#0066ff')
	rects2 = ax.bar(x + width/2, test_dist3, width, label='Test (n='+n_test+')', color='#9966ff')
	plt.semilogy()
	
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Count')
	ax.set_xlabel('Taxon (approximately phylum)')
	ax.legend()
	plt.xlim(-.4,len(labels)-.7)	
	
	fig.tight_layout()
	
	out = ''
	for i, lab in enumerate(y3.index.tolist()):
	    out = out+str(i)+": "+lab+", "
	out
	
	return fig, out, y3





def yucko_genome_heatmap2(corrupted_test, idx, model, all_kos):
	from matplotlib.colors import LinearSegmentedColormap
	import sklearn as sk
	
	n_features = int(corrupted_test.shape[1]/2)
	
	mod_to_kos = {}
	for org in org_to_mod_to_kos:
	    mods = org_to_mod_to_kos[org]
	    
	    for mod in mods:
	        if mod not in mod_to_kos:
	            mod_to_kos[mod] = mods[mod]
	
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
	
	proc_to_ko_idx = {}
	for proc in process_to_mod:
	    for mod in process_to_mod[proc]:
	        if mod in mod_to_kos:
	            kos = mod_to_kos[mod]
	            idxs = []
	            for ko in kos:
	                idx = all_kos.index(ko)
	                idxs.append(idx)
	            if proc in proc_to_ko_idx:
	                proc_to_ko_idx[proc].extend(idxs)
	            else:
	                proc_to_ko_idx[proc] = idxs

	procs = proc_to_ko_idx.keys()
	
	idx_order = []
	for proc in procs:
	    idx_order.extend(proc_to_ko_idx[proc])
	idx_order = list(set(idx_order))
	
	for i in range(n_features):
	    if i not in idx_order:
	        idx_order.append(i)
	
	idx = 1
	n_extension = 100*100 - n_features
	# get corrupted genomes
	corrupted_genome = corrupted_test[idx, :n_features]
	corrupted = torch.reshape(corrupted_genome, (1, n_features))
	corrupted = corrupted[:,idx_order].tolist()[0]
	corrupted.extend([4] * n_extension) # 100*100 - n_features
	corrupted = np.reshape(corrupted, (100, 100))  
	
	# Get uncorrupted version of genome
	uncorrupted_genome = corrupted_test[idx, n_features:]
	uncorrupted = torch.reshape(uncorrupted_genome, (1, n_features))
	uncorrupted = uncorrupted[:,idx_order].tolist()[0]
	uncorrupted.extend([4] * n_extension) # 100*100 - n_features
	uncorrupted = np.reshape(uncorrupted, (100, 100))  
	
	# Get predicted uncorrupted version of genome
	model.eval()
	pred = model.forward(corrupted_genome)
	pred = pred[0].tolist()
	binary_pred = [1 if i > 0.5 else 0 for i in pred]
	
	colours = ['black', 'green', 'orange', 'yellow', 'white']
	cmap_name = 'my_list'
	
	n_features = int(corrupted_test.shape[1]/2)
	n_extension = 100*100 - n_features
	cm = LinearSegmentedColormap.from_list(cmap_name, colours, N=len(colours))
	
	colour_pred = []
	for i in zip(binary_pred, corrupted_genome, uncorrupted_genome):
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
	colour_pred = np.reshape(colour_pred, (100, 100))  
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
	ax1.imshow(uncorrupted, cmap=cm, interpolation='nearest')
	ax2.imshow(corrupted, cmap=cm, interpolation='nearest')  
	ax3.imshow(colour_pred, cmap=cm, interpolation='nearest')
	ax1.set_title("Original (uncorrupted)")
	ax2.set_title("Corrupted")
	ax3.set_title("Generated")

	return fig

def mods_by_genomes(org_to_mod_to_kos):	
	
	n_genomes = len(org_to_mod_to_kos)
	_ = [list(org_to_mod_to_kos[i].keys()) for i in org_to_mod_to_kos]
	n_mods = len(list(set([item for sublist in _ for item in sublist])))
	
	fig = plt.figure()
	plt.hist([len(org_to_mod_to_kos[i]) for i in org_to_mod_to_kos], 50)
	plt.xlabel("Number of modules per genome")
	plt.ylabel("Frequency")
	plt.title("Distribution of the # of modules (n="+str(n_mods)+") per genome (n="+str(n_genomes)+")")

	return fig















