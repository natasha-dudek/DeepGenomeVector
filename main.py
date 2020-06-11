from argparse import Namespace

import torch

from genome_autoencoder import models
from genome_autoencoder import util 

flags = Namespace(
    DATA_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_autoencoder/data/pangenome_matrix_t0.tab', 
    SAVE_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_autoencoder',
    MALLET_PATH = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/mallet-2.0.8',
    n_test = 0.1, # train-test split, n * 100 = % of data that goes into test set (e.g.: 0.1 -> 10%)
    ngeltons = 10, # drop singleton (n-gletons) clusters from analysis
    num_epochs = 20,
    batch_size = 10,
    lr = 1e-3,
    print_every = 5, # print loss every n batches during training 
    replacement_threshold = 0.5, # probability over which binarizer converts to a 1
    cooccur_measure = 'ami' # [ami | spearman], ami = adjusted mutual information
)

torch.manual_seed(0)

# load data, create dataloader, get some basic stats
print('Loading dataset')
train_dl, test_dl, train_data, test_data, num_clusters, num_genomes, cluster_names = util.prepare_data(flags.DATA_FP, flags.ngeltons, flags.n_test, flags.batch_size)

# generate co-occurence matrix for all gene clusters (or load from file)
print('Generating co-occurence matrix for all gene clusters (or loading from file)')
co_occur_matrix = util.calc_co_occurence(flags.SAVE_FP, flags.cooccur_measure, num_genomes, num_clusters)

# define the network
print('defining network')
model = models.AutoEncoder(num_clusters)
print(model)