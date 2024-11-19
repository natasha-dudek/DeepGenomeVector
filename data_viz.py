import math
import random
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
import seaborn as sns
import sklearn as sk
import torch
from scipy import interp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import auc, roc_curve


def calc_svd(data, n_components):
	"""Perform truncated SVD on data.

	Note: SVD works efficiently on sparse matrices, unlike PCA

	Arguments:
			data (tensor) -- input data
			n_components (int) -- desired dimensionality of output data

	Returns:
			svd_result (np.array) -- SVD results with reduced dimensionality
	"""

	# First determine how many components to use (upper bound of 50 = max for input to tSNE)
	# I.e. as many as it takes to capture 99% of the variance or 50, whichever is lower
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


def my_roc_curve(target, y_probas):
	"""Performs ROC / AUC calculations and plots ROC curve.

	Arguments:
			target (numpy.ndarray) -- uncorrupted version of genomes (n_genomes, n_features)
			y_probas (numpy.ndarray) -- prediction for corrupted genomes (n_genomes, n_features)

	Returns:
			matplotlib.Figure
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
			continue

	# Calculate micro-average
	fpr_micro, tpr_micro, _ = roc_curve(target.ravel(), y_probas.ravel())
	roc_auc["micro"] = auc(fpr_micro, tpr_micro)

	n_examples = 100  # will plot 50 example genes on ROC curve

	# get colours for plotting
	cm = plt.cm.get_cmap("gist_rainbow")
	c = np.linspace(0, 1, 50)  # start, stop, how_many
	colours = [cm(i) for i in c]
	colours = colours * 2

	# plot
	fig, ax = plt.subplots(figsize=(5, 5))
	a = random.sample(range(target.shape[1]), n_examples)
	for i in range(len(a)):
		plt.plot(fpr[a[i]], tpr[a[i]], color=colours[i], alpha=0.5, lw=1)
	plt.plot(
		fpr_micro,
		tpr_micro,
		color="black",
		lw=2,
		label="Micro-average (AUC = %0.2f)" % roc_auc["micro"],
	)
	plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	# plt.title('ROC for 50 randomly selected genes + micro-average')
	plt.legend(loc="upper right")

	return fig


def plot_tax_dist(c_train_genomes, c_test_genomes, train_tax_dict, test_tax_dict):
	"""Plot histogram showing the number of genomes per phylum, for both the
	training and test sets.

	Arguments:
			c_train_genomes (list) -- names of genomes in the order they appear in the corrupted training dataset
			c_test_genomes (list) -- names of genomes in the order they appear in the corrupted test dataset
			train_tax_dict (dict) -- maps each training set tla to ['domain', 'phylum', ..., 'species']
			test_tax_dict (dict) -- maps each test set tla to ['domain', 'phylum', ..., 'species']

	Returns:
			matplotlib.Figure
	"""
	blue = "#7094db"
	red = "#990000"	
	
	n_train = str(int(len(c_train_genomes) / 100))
	n_test = str(int(len(c_test_genomes) / 100))

	def phyla(tax_dict):
		dist = []
		for i in tax_dict:
			phylum = tax_dict[i][1]
			if phylum == "Proteobacteria":
				phylum = tax_dict[i][2]
			dist.append(phylum)

		dist2 = pd.DataFrame({"phylum": dist})
		dist2["freq"] = ""
		y = dist2.groupby("phylum").count()  # .reset_index()

		return y

	y1 = phyla(train_tax_dict)
	y2 = phyla(test_tax_dict)

	# Get labels
	y3 = pd.merge(y1, y2, how="outer", left_index=True, right_index=True)
	y3["freq_y"] = y3["freq_y"].fillna(0)
	y3 = y3.sort_values("freq_x", ascending=False)
	train_dist3 = y3["freq_x"].tolist()
	test_dist3 = y3["freq_y"].tolist()
	labels = [i for i in range(len(train_dist3))]

	x = np.arange(len(labels))  # the label locations
	width = 0.35  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(
		x - width / 2,
		train_dist3,
		width,
		label="Train (n=" + n_train + ")",
		color=blue,
	)
	rects2 = ax.bar(
		x + width / 2,
		test_dist3,
		width,
		label="Test (n=" + n_test + ")",
		color=red,
	)
	plt.semilogy()

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel("Count")
	ax.set_xlabel("Taxon (approximately phylum)")
	ax.legend()
	plt.xlim(-0.4, len(labels) - 0.7)

	fig.tight_layout()

	return fig


def mods_by_genomes(tla_to_mod_to_kos):
	"""Plot a histogram showing the distribution of the number of modules per
	genome.

	Arguments:
			tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})

	Returns:
			matplotlib.Figure
	"""
	n_genomes = len(tla_to_mod_to_kos)
	_ = [list(tla_to_mod_to_kos[i].keys()) for i in tla_to_mod_to_kos]
	n_mods = len(list(set([item for sublist in _ for item in sublist])))

	fig = plt.figure()
	plt.hist([len(tla_to_mod_to_kos[i]) for i in tla_to_mod_to_kos], 50)
	plt.xlabel("Number of modules per genome")
	plt.ylabel("Frequency")
	plt.title(
		"Distribution of the # of modules (n="
		+ str(n_mods)
		+ ") per genome (n="
		+ str(n_genomes)
		+ ")"
	)
	plt.xlim(left=0)

	return fig


def variants_of_mod(mod, mod_sets, tla_to_mod_to_kos):
	"""Generate a histogram showing the variants of a module across all genomes
	in the full dataset (remember that for training the actual model we only
	use the most common version of the module)

	Arguments:
			mod (str) -- the name of a module (e.g.: 'M00001')
			mod_sets (defaultdict) -- raw data from KEGG defining which KOs are in each module
			tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})

	Returns:
			matplotlib.Figure
	"""
	mod_set = list(mod_sets[mod].values())
	mod_set.sort(reverse=True)

	fig = plt.figure()
	x_labels = [i for i in range(len(mod_sets[mod].values()))]
	plt.bar(x_labels, mod_set)
	plt.yscale("log")
	plt.title("Variants of module " + mod)
	plt.xlabel("Variant (n=" + str(len(mod_sets[mod].values())) + " )")
	plt.ylabel("Count across all genomes (n=" + str(len(tla_to_mod_to_kos)) + ")")
	plt.xlim(left=0)
	print(
		"The most common variant of module "
		+ mod
		+ " occurs in %s genomes. There are a total of %s module variants."
		% (max(mod_sets[mod].values()), len(mod_set))
	)

	return fig


def kos_per_genome(tnum_to_kos, train_genomes, test_genomes):
	"""Plot the number of KOs encoded by each genome in the full dataset and
	print summary stats.

	Note: this includes genomes excluded from the final dataset

	Arguments:
			tnum_to_kos (dict) -- maps tnums to KOs encoded by that genome
			train_genomes (list) -- tnums of genomes in the training set
			test_genomes (list) -- tnums of genomes in the test set

	Returns:
			matplotlib.Figure
	"""
	fig = plt.figure()
	plt.hist([len(tnum_to_kos[i]) for i in tnum_to_kos], 50)
	plt.xlabel("Number of KOs per genome")
	plt.ylabel("Frequency")
	plt.title("Histogram of KOs per genome")
	plt.xlim(left=0)
	lens = [
		len(tnum_to_kos[i])
		for i in tnum_to_kos
		if i in train_genomes or i in test_genomes
	]
	print("Median:", np.median(lens), "Min:", min(lens), "Max:", max(lens))

	return fig


def distrib_num_genomes_with_mod(tla_to_mod_to_kos):
	"""Count and plot the number of genomes that encode each module.

	Arguments:
			tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})

	Returns:
			matplotlib.Figure
	"""
	mods_count = defaultdict(int)
	org_count = 0
	for org in tla_to_mod_to_kos:
		for mod in tla_to_mod_to_kos[org]:
			mods_count[mod] += 1
		org_count += 1

	fig = plt.figure()
	plt.hist(mods_count.values(), 10)
	plt.xlabel("Number of genomes encoding each module")
	plt.ylabel("Frequency")
	plt.title(
		"Distribution of the # of genomes (n="
		+ str(org_count)
		+ ") encoding each module (n="
		+ str(len(mods_count))
		+ ")"
	)
	plt.yscale("log")
	plt.xlim(left=0)

	print(
		"Number of mods encoded in only one genome:", list(mods_count.values()).count(1)
	)
	print("Max number of genomes encoding a single mod", max(list(mods_count.values())))

	return fig


def perc_genes_in_mods(tla_to_mod_to_kos, tnum_to_kos, tla_to_tnum, all_kos):
	"""Plots a histogram of the percentage of genes per genome that contribute
	to modules.

	Arguments:
			tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
			tnum_to_kos (dict) -- maps tnums to KOs encoded by that genome, e.g.: 'T00001': [K00001, ... 'K0000N']
			tla_to_tnum (dict) -- maps tlas to tnums
			all_kos (list) -- all KOs in the entire dataset

	Returns:
			matplotlib.Figure
	"""
	mods = []  # will be used to calculate the total number of mods in the dataset
	perc_genes_in_mods = []
	for tla in tla_to_mod_to_kos:
		if tla not in tla_to_mod_to_kos:
			continue
		if tla_to_tnum[tla] not in tnum_to_kos:
			continue

		# get total number of genes in current genome
		n_genes = len(tnum_to_kos[tla_to_tnum[tla]])

		# get total number of unique genes that contribute to modules in the current genome
		genes_in_mods = []
		for mod in tla_to_mod_to_kos[tla]:
			genes_in_mods.extend(tla_to_mod_to_kos[tla][mod])
			mods.append(mod)
		n_genes_in_mods = len(list(set(genes_in_mods)))

		# get percentage of genes contributing to modules
		try:
			perc_genes_in_mods.append(n_genes_in_mods / n_genes * 100)
		except (
			ZeroDivisionError
		):  # there is a bad data pt that haven't been filtered out yet
			continue

	n_mods = len(list(set(mods)))

	fig = plt.figure()
	plt.hist(perc_genes_in_mods, 100)
	plt.xlabel("Percent of KOs contributing to modules")
	plt.ylabel("Frequency")
	plt.title(
		"Distribution of the % of KOs (n="
		+ str(len(all_kos))
		+ ") represented by modules (n="
		+ str(n_mods)
		+ ") per genome (n="
		+ str(len(tla_to_mod_to_kos))
		+ ")"
	)
	plt.xlim(left=0)

	# Calculate some summary statistics
	kos_in_mods = []

	for tla in tla_to_mod_to_kos:
		for mod in tla_to_mod_to_kos[tla]:
			kos_in_mods.extend(tla_to_mod_to_kos[tla][mod])

	kos_in_mods = list(set(kos_in_mods))

	print("The total number of genes that occur in mods is", len(kos_in_mods))
	print("The total number of genes in the dataset is", len(all_kos))
	print(
		"Across the full dataset, the % of KOs that contribute to mods is",
		str(round(len(kos_in_mods) / len(all_kos) * 100, 2)) + "%",
	)

	return fig


def genes_per_genome(c_train_genomes, c_test_genomes, tnum_to_kos, tla_to_tnum):
	"""Plot a histogram showing the number of genes per genome.

	Arguments:
			c_train_genomes (list) -- names of genomes in the order they appear in the corrupted training dataset
			c_test_genomes (list) -- names of genomes in the order they appear in the corrupted test dataset
			tnum_to_kos (dict of lists) -- for each organism, a list of KOs encoded by the genome
			tla_to_tnum (dict) -- maps tla to tnum for each genome

	Returns:
			matplotlib.Figure
	"""
	blue = "#7094db"
	red = "#990000"	
	
	# Get lengths of each genome
	def get_lengths(c_genomes, tnum_to_kos):
		lens = []
		for i in list(set(c_genomes)):
			tnum = tla_to_tnum[i]
			lens.append(len(tnum_to_kos[tnum]))
		return lens

	train_lens = get_lengths(c_train_genomes, tnum_to_kos)
	test_lens = get_lengths(c_test_genomes, tnum_to_kos)

	# Plot the number of KOs encoded by each genome
	fig, ax = plt.subplots()
	plt.hist(train_lens, 50, color=blue)
	plt.hist(test_lens, 50, color=red)
	ax.legend(["Train", "Test"])
	plt.xlabel("# annotated genes / genome")
	plt.ylabel("Count")
	plt.ylim(0, 125)
	plt.xlim(min(train_lens), max(train_lens) + 10)

	all_lens = train_lens + test_lens
	print("Min # genes:", min(all_lens))
	print("Median # genes:", np.median(all_lens))
	print("Max # genes:", max(all_lens))

	return fig


def mods_per_train_genome(tla_to_mod_to_kos, c_train_genomes):
	"""Plot histogram showing the number of modules per training genome.

	Arguments:
			tla_to_mod_to_kos (defaultdict of dicts) -- maps tla to series of dicts, keys are KEGG modules and values are lists of KOs in that module (e.g.: 'eun': {'M00001': ['K00845', etc]}, etc} etc})
			c_train_genomes (list) -- names of genomes in the order they appear in the corrupted training dataset

	Return:
			matplotlib.Figure
	"""
	mods_count = defaultdict(int)
	for org in tla_to_mod_to_kos:
		if org in c_train_genomes:
			mods_count[org] = len(tla_to_mod_to_kos[org])

	fig = plt.figure()
	plt.hist(mods_count.values())
	plt.xlabel("Number of modules")
	plt.ylabel("Number of genomes")
	plt.xlim(left=0)
	return fig
