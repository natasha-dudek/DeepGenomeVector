import torch
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np

def genome_to_gensim(test_data, cluster_names):
	"""
	Prepare genome data for working with gensim
	
	word = gene cluster
	document = genome
	corpus = collection of genomes
	
	Arguments:
	test_data -- data on which to perform LDA (torch.utils.data.dataset.Subset)
	cluster_names -- names of gene clusters (list)
	
	Returns:
	clustr_corpus -- gensim-style corpus for text (list)
	clustr_dictionary -- gensim dictionary for text (gensim.corpora.dictionary.Dictionary)
	"""
	
	# convert type(test_data) -> numpy array
	y = torch.tensor([i.numpy() for i in test_data]).float().numpy()
	
	
	# Create dicts converting numerical ID [0 - (# clusters - 1)]
	# gene clusters look something like 825_ADC48894.1
	id2cluster = {}
	for i in range(len(cluster_names)):
	    id2cluster[i] = cluster_names[i]
	
	# Create "texts" -- a list of lists. Each sublist represents a genome and is a list of all the genes it contains
	# gene clusters look something like 825_ADC48894.1
	texts = []
	for i in y:
	    tempy = []
	    for s in range(len(i)):
	        if i[s] == 1:
	            tempy.append(id2cluster[s])
	    texts.append(tempy)

	clustr_dictionary = Dictionary(texts)
	clustr_corpus = [clustr_dictionary.doc2bow(text) for text in texts]
	
	return clustr_corpus, clustr_dictionary, texts, id2cluster

def perform_lda(texts, clustr_corpus, clustr_dictionary, coherence_measure, num_iters, k):
	
	"""
	Performs LDA
	
	Arguments: 
	texts -- list of lists. Each list represents a genome by which genes it has present (e.g. [1233_ADC49302.1, 1250_ADC49319.1, 1197_ADC49266.1])
	clustr_corpus -- gensim-style corpus for text (list)
	clustr_dictionary -- gensim dictionary for text (gensim.corpora.dictionary.Dictionary)
	coherence_measure -- method to calculate topic coherence score ["u_mass" | "c_v"]
	num_iters -- max number of iterations through the corpus when inferring the topic distribution of a corpus
	k -- number of topics
		
	Returns:
	lda_model - fitted LDA model (gensim.models.ldamodel.LdaModel)
	cm - coherence model (type is gensim.models.coherencemodel.CoherenceModel)
	"""
	
	
	lda_model = LdaModel(corpus=clustr_corpus, id2word=clustr_dictionary, iterations=num_iters, num_topics=k)
	
	if coherence_measure == "u_mass":
		cm = CoherenceModel(model=lda_model, corpus=clustr_corpus, dictionary=clustr_dictionary, coherence=coherence_measure)
	elif coherence_measure == "c_v":
		cm = CoherenceModel(model=lda_model, texts=texts, dictionary=clustr_dictionary, coherence=coherence_measure)
	
	return lda_model, cm


def lda_elbow(texts, clustr_corpus, clustr_dictionary, coherence_measure, start_k, stop_k, step_k, num_iters):
	
	"""
	Select number of topics for LDA, return according LDA model
	
	Arguments:
	texts -- list of lists. Each list represents a genome by which genes it has present (e.g. [1233_ADC49302.1, 1250_ADC49319.1, 1197_ADC49266.1])
	clustr_corpus -- gensim-style corpus for text (list)
	clustr_dictionary -- gensim dictionary for text (gensim.corpora.dictionary.Dictionary)
	coherence_measure -- method to calculate topic coherence score ["u_mass" | "c_v"]
	start_k -- smallest # of clusters to evaluate
	stop_k -- largest # of clusters to evaluate
	step_k -- step size by which to increase # clusters from start_k to stop_k
	num_iters -- max number of iterations through the corpus when inferring the topic distribution of a corpus
	
	Returns:	
	"""

	k_vals = []
	lda_models = []
	coherences = []
	
	for k in range(start_k, (stop_k+step_k), step_k):
		print("performing LDA, calculating coherence score for k =",k)
		lda_model, cm = perform_lda(texts, clustr_corpus, clustr_dictionary, coherence_measure, num_iters, k)
		k_vals.append(k)
		lda_models.append(lda_model)
		coherences.append(cm.get_coherence())
	
	return k_vals, lda_models, coherences

def plot_lda_elbow(k_vals, coherences, SAVE_FP):
	
	fig=plt.figure()
	ax=fig.add_subplot(111)
	
	ax.plot(k_vals, coherences, marker='o', c='b', fillstyle='none')
	plt.title("Coherence score vs number of topics")
	plt.ylabel("Coherence score")
	plt.xlabel("Number of topics")
	#plt.axvline(x=900)
	plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.5)
	plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.5)
	plt.savefig(SAVE_FP+"/lda_elbow.pdf")
	