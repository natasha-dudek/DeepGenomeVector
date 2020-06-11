import numpy as np
import pandas as pd
import time
import os
import igraph as ig
import sys

def calc_cooccurence_matrix(df, method):

    """
    Calculate co-occurence matrix
    
    Arguments:
    df -- pandas df containing rows = genomes, columns = gene clusters. Entries are expected to be 0 or 1.
    method -- co-occurence metric to use ["ami" | "sum" | "spearman"]
        ami = adjusted mutual information (note: due to adjustment, can have negative numbers)
        count = number of times two gene clusters both occur within a genome over whole set of genomes
        spearman = only plot vals with p-val <= 0.05 (post-Bonferroni correction)
    Returns:
    cooccur_matrix -- np array of co-occurences, dtype = int by default, possibly replaced by float dep on coocur metric
    columns -- headers / columns for cooccur_matrix (i.e. labels) *** 0, 1, 2, 3, 4, 5, ... , n_columns
    """
    
    # Verify that pandas df meets expectation that all entries are 0's or 1's
    list_df = df.values.tolist()
    flat_list = [item for sublist in list_df for item in sublist]
    flat_list = list(set(flat_list))
    error_flag = False
    for i in flat_list: 
        if (i!=1) and (i!=0): 
            error_flag = True
    if error_flag:
        print("ERROR: Your pandas df must contain only 0's and 1's.")
        sys.exit(1)
    
    # Calculate co-occurence matrix
    columns = list(df)
    n_clusters = len(columns)
    cooccur_matrix = np.zeros((n_clusters, n_clusters), dtype=float)
    
    for i in range(0,(n_clusters)):
        for j in range(0,(n_clusters)):
            if method == "ami":
                # Measure of co-occurence = adjusted mutual information score (scikit-learn)
                cooccur_matrix[i,j] = adjusted_mutual_info_score(df[columns[i]], df[columns[j]])
            elif method == "spearman":
                # spearman's ranges from -1 to +1
                rho, pval = spearmanr(df[columns[i]],df[columns[j]], axis=1)
                if pval <= 0.05/(n_clusters*n_clusters):
                    cooccur_matrix[i,j] = rho
                else:
                    cooccur_matrix[i,j] = 0
            elif method == "count":
                # count how often two gene clusters are present within a single genome
                x = df[columns[i]] + df[columns[j]] # if sum >1, both are present
                cooccur_matrix[i,j] = x[x>1].shape[0] # count the number of genomes for which this condition is met
                cooccur_matrix[i,j] = cooccur_matrix[i,j].astype(int)
    
    return cooccur_matrix, columns

def cooccurence_matrix(SAVE_FP, cooccur_measure, num_genomes, num_clusters, data):
    """
    Determines whether to load pre-existing co-occurence matrix or generate one de novo (slow)
    
    Arguments:
    SAVE_FP -- where to check for a pre-existing co-occurence matrix / where to save matrix if generating de novo
    cooccur_measure -- co-occurence metric to use ["ami" | "sum" | "spearman"]
        ami = adjusted mutual information (note: due to adjustment, can have negative numbers)
        count = number of times two gene clusters both occur within a genome over whole set of genomes
        spearman = only plot vals with p-val <= 0.05 (post-Bonferroni correction)
    num_genomes -- number of genomes in dataset
    num_clusters -- number of gene clusters in dataset
    data -- torch tensor where rows = genome, columns = gene clusters
    
    Returns:
    cooccur_mat -- co-occurence matrix, np array with dtype float. No header, no row labels
    cooccur_cols -- Header / row labels for co-occurence matrix
    """
    
    
    # Create co-occurence matrix
    # This is slow
    cooccur_mat_fp = SAVE_FP+'cooccur_mat_'+cooccur_measure+'_'+str(num_genomes)+'_'+str(num_clusters)+'.csv'
    cooccur_cols_fp = SAVE_FP+'cooccur_cols_'+cooccur_measure+'_'+str(num_genomes)+'_'+str(num_clusters)+'.csv'
    
    if os.path.isfile(cooccur_mat_fp) and os.path.isfile(cooccur_cols_fp):
        print("Co-occurence matrix already exists, loading from file")
        cooccur_mat = pd.read_csv(cooccur_mat_fp, sep=',',header=None).values
        cooccur_cols = pd.read_csv(cooccur_cols_fp, sep=',',header=None).values
        
    else:
        print(("No precomputed co-occurence matrix is available in %s, generating one from scratch using %s") % (SAVE_FP, cooccur_measure))
        start_time = time.time()
        
        # convert torch tensor data to df
        df = pd.DataFrame(data.numpy())
        
        cooccur_mat, cooccur_cols = calc_cooccurence_matrix(df, cooccur_measure)
        np.savetxt(cooccur_mat_fp, cooccur_mat, delimiter=',')
        np.savetxt(cooccur_cols_fp, cooccur_cols, delimiter=',')
        print("Time to calculate co-occurence matrix: %s hours" % ((time.time() - start_time)/60/60))
    
    return cooccur_mat, cooccur_cols, cooccur_mat_fp    
    
def adjmat_to_igraph(cooccur_mat, co_occur_columns, direction):
	"""
	Converts adjacency matrix (e.g.: co-occurrence matrix) into a igraph graph 
	
	Assumption: nodes = gene clusters, edges = co-occurence of gene clusters
	
	Arguments:
	cooccur_mat -- adjacency matrix in the form of a numpy array
	direction -- is this a directed or undirected graph? ["directed" | "undirected"]
	
	Returns:
	g -- an igraph graph, type igraph.Graph
	"""
	if direction == "directed":
		dir_num = 0
	elif direction == "undirected":
		dir_num = 1
	else:
		print ("Your only options are directed or undirected. Please try again")
			 
	# Cannot pass a numpy array as an adjacency matrix to igraph, first convert from numpy to list using tolist()
	# Ints in an adjacency matrix are interpreted by igraph as the # of edges between nodes rather than the weights of the edges
	# To solve this problem, use boolean of adjacency matrix to create graph (0 = no edge, >0 = edge)
	g = ig.Graph.Adjacency((cooccur_mat > 0).tolist(), direction)
	
	# Add edge weights and node labels using on all non-zero counts in the adjacency matrix
	g.es['weight'] = cooccur_mat[cooccur_mat.nonzero()]
	g.vs['label'] = co_occur_columns
	
	return g

def igraph_stats(g):

	print("Is the graph directed:", g.is_directed())
	print("Are graph edges weighted:", g.is_weighted())
	print("Number of vertices in the graph:", g.vcount())
	print("Number of edges in the graph", g.ecount())
	print("Maximum degree in the graph:", g.maxdegree())
	print("Highest value weight:", max(g.es['weight']))


def normalize_coocurrence(co_occur_counts):
	"""
	Normalize co-occurence counts
	
	Let's say one gene cluster A is present in 100% of genomes while cluster B is present in only 10%
	We want to take into account that the co-occurence is asymmetrical
	This function divides the count of cooccurences for each gene A : gene B by the # of occurences of gene A, etc
	
	Arguments:
	co_occur_counts -- co-occurence matrix generated using counts
	
	Returns:
	norm_coocur -- co-occurence matrix where values have been normalized to be fractions (range 0-1)
	"""
	
	
	norm_coocur = np.zeros(shape=(len(co_occur_counts),len(co_occur_counts)))
	
	for i in range(len(co_occur_counts)): # for every row
		
		normalized_row = np.zeros(shape=(len(co_occur_counts),))
		num_occurence = co_occur_counts[i][i] # for a given gene cluster, this is how many times it occured across all genomes
		
		for j in range(len(co_occur_counts)): # for every column
			if num_occurence == 0: 
				normalized_row[j] = 0
			else: normalized_row[j] = co_occur_counts[i][j] / num_occurence
		
		norm_coocur[i] = normalized_row

	return norm_coocur






    
    
    
    