"""
Module that loads data from disk.

This takes a long time, so we want to avoid doing it more than once. To gain data access,
simply import X and Y from this module, and python's module import rules will make sure the
loading only gets run once.
"""

import os

import torch

from genome_embeddings import config

train_data = torch.load(os.path.join(config.TRAIN_DATA_PATH))
test_data = torch.load(os.path.join(config.TEST_DATA_PATH))

print("loaded train + test data")

# To make predictions on (ROC + AUC)
num_features = int(train_data.shape[1]/2)
corrupt_test_data = test_data[:,:num_features]

X = train_data[:,:num_features]  #corrupted genomes in first half of matrix columns
y = train_data[:,num_features:]  #uncorrupted in second half of matrix columns