# DeepGenome

This repository contains a python implementation (tested on version 3.7.6) for the DeepGenome model described in: 
Dudek, N.K. & Precup, D. Towards AI-designed genomes using machine learning. In prep for submission (2022).

Here, we introduce a framework for training a machine learning algorithm to learn the basic genetic principles underlying the gene composition of bacterial genomes. Given a set of desired pathways (e.g., glycolysis, TCA cycle, and butane degradation), our variational autoencoder (VAE) model strives to generate synthetic bacterial "genome vectors" - vectors denoting the full complement of genes that would need to be encoded to support a viable cell that supports the user-defined input functions.

## Installation

pip install requirements.txt

## How to run

The core module is main.ipynb. This juptyer notebook is used to load the data, process and filter it, perform data exploration, train a single VAE model or load a trained model (i.e. the best model resulting from hyperparameter optimization), and then analyze the results and generate the figures seen in the manuscript. 

To perform hyperparameter tuning, use the hpo_on_cluster.py module. Note that it is recommend to do this on an HPC cluster due to significant compute requirements. 

Data was sourced from the [KEGG database](https://www.genome.jp/kegg/). A substantial amount of processing was performed on the data prior to input into the model. You can access processed training data here and test data here. 

