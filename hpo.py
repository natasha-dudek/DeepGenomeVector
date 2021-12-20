from argparse import Namespace
from collections import defaultdict
from datatime import date
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle
from ray import tune
import torch

from genome_embeddings import trainable # must import before ray

import ray # must import after trainable

date = date.today()
BASE_DIR = '/home/ndudek/projects/def-dprecup/ndudek/'

# Set experimental parameters
settings = Namespace(
	DATA_FP = BASE_DIR+'hp_tuning_'+date+'/',
	SAVE_FP = BASE_DIR+'hp_tuning_'+date+'/',
	num_epochs = 10,
	num_cpus=10, 
	replacement_threshold = 0.5, # probability over which binarizer converts to 
	a 1
	num_corruptions = 100, # number of corrupted versions of a genome to produce
	)
    
# Initialize ray tune
memory = 2000 * 1024 * 1024
object_store_memory = 200 * 1024 * 1024
driver_object_store_memory=100 * 1024 * 1024
ray.shutdown()
ray.init(local_mode=True, memory=memory, 
        object_store_memory=object_store_memory,
        driver_object_store_memory=driver_object_store_memory,
        num_cpus=10)

# Set config file parameters
config = {'num_epochs': 10,
         'kfolds': 10,
         'replacement_threshold': settings.replacement_threshold,
         'nn_layers': tune.choice([1, 2, 3, 4]),
         'batch_size': tune.choice([32, 64, 128, 256]),
          'lr': tune.loguniform(1e-4, 1e-1), 
          'weight_decay': tune.loguniform(1e-5, 1e-2) 
         }

# Perform hp tuning
analysis = tune.run(
    trainable.train_AE, 
    name='vae'+date,
    config=config,
    verbose=2, 
    resources_per_trial={
            'cpu': 10,
            'gpu': 0
    },
    num_samples=100,  
    queue_trials=True,
    #local_dir='/Users/natasha/Desktop/TUNE_RESULT_DIR',
    local_dir=BASE_DIR+'hp_tuning_'+date+'//TUNE_RESULT_DIR'
    )

print('Best config is:', analysis.get_best_config(metric='test_f1'))
analysis.get_best_config(metric='test_f1')