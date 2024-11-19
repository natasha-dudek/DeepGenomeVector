import argparse
import gc
import os
import random
import sys
import time
from argparse import Namespace
from copy import deepcopy

import numpy as np
import psutil

np.seterr(all="raise")
import pickle

import pandas as pd
import ray
import sklearn as sk
import torch
from filelock import FileLock
from ray import tune
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Binarizer

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from genome_embeddings import config, data, models


def binarize(pred_tensor, replacement_threshold):
    """Model predictions are converted from y_probas to concrete 1's or 0's.

    Arguments:
            pred_tensor (numpy.ndarray) -- model predictions
            threshold (float) -- threshold at which to replace pred scores with 0 pr 1

    Returns:
            binary_preds (list of numpy.ndarray) -- binarized predictions
    """

    binary_preds = []
    for i in pred_tensor:
        try:
            pred_arr = i.detach().numpy()
        except AttributeError:
            pred_arr = i
        b = Binarizer(threshold=replacement_threshold).fit_transform(
            pred_arr.reshape(1, -1)
        )
        binary_preds.extend(b)

    return binary_preds


def f1_score(pred_non_bin, target, replacement_threshold):
    """Calculates F1 score.

    Arguments:
            pred_non_bin (tensor) -- predictions output by model (probability values)
            target (tensor) -- Uncorrupted genomes (binary)
            replacement_threshold (float) -- threshold at which to replace pred scores with 0 or 1

    Returns:
            f1 (float) -- average F1 score for batch
    """
    binarized_preds = binarize(pred_non_bin, replacement_threshold)

    f1s = []
    recalls = []
    precisions = []

    for i in range(0, len(binarized_preds)):
        f1 = sk.metrics.f1_score(
            target.data.numpy()[i], binarized_preds[i], zero_division=0
        )
        f1s.append(f1)

    f1 = sum(f1s) / len(f1s)

    return f1


def cv_dataloader(batch_size, num_features, k):
    """
    Creates training dataloader w/ k-folds for cross validation
    Note: only creates 1 split regardless of # k -- improves training speed (method of skorch CVSplit)

    Arguments:
            batch_size (int) -- batch size for training set
            num_features (int) -- number of features / genes
            k (int) -- number of folds

    Returns:
            dict of DataLoaders -- train and cross-validation dataloaders
                    dict["train"] -- training dataloader, batch_size = batch_size
                    dict["cv"] -- cross-validation dataloader, batch_size = 1000 (hard-coded)
    """
    # load train data from memory (saves time and, more importantly, space)
    X = data.X  # corrupted genomes
    y = data.y  # uncorrupted genomes
    
    # THIS IS HACKY
    # when the data was generated, we did 100 corruptions for each genome,
    # so we know a new genome starts every 100 lines. TODO: pass through
    # genome ID or split earlier
    
    idx_genome_start = [i for i in range(0, len(X), 100)]
    num_cv = len(idx_genome_start) // k
    cv_idx_start = np.random.choice(idx_genome_start, num_cv, replace=False)
    train_idx_start = list(set(idx_genome_start) - set(cv_idx_start))
    cv_idx = [i+j for i in cv_idx_start for j in range(100)]
    train_idx = [i+j for i in train_idx_start for j in range(100)]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_cv = X[cv_idx]
    y_cv = y[cv_idx]

    
#
#    # Create random split for 1-time k-fold validation
#    idx_genomes = [i for i in range(len(X))]
#    num_cv = int(len(idx_genomes) / k)
#    num_train = len(idx_genomes) - num_cv
#    cv_idx = np.random.choice(idx_genomes, num_cv, replace=False)
#    train_idx = list(set(idx_genomes) - set(cv_idx))
#    X_train = X[train_idx]
#    y_train = y[train_idx]
#    X_cv = X[cv_idx]
#    y_cv = y[cv_idx]

    # Create dataloaders
    batch_size_cv = 1000
    train_ds = TensorDataset(X_train, y_train)
    cv_ds = TensorDataset(X_cv, y_cv)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, drop_last=False, shuffle=True
    )
    cv_dl = DataLoader(cv_ds, batch_size=batch_size_cv, shuffle=True)

    return {"train": train_dl, "cv": cv_dl}


def cv_vae(model, loaders, replacement_threshold, device=torch.device("cpu")):
    """Evaluate model on cross-validation set.

    Arguments:
            model (genome_embeddings.models.VariationalAutoEncoder) -- pytorch model
            loaders (dict of DataLoaders) -- dictionary of dataloader, here we use loaders["cv"]
            replacement_threshold (float) -- threshold for converting predicted probabilities to 1's or 0's
            device (str) -- cpu or cuda

    Returns:
            loss (float) -- Loss on a randomly selected batch from the test set
            f1 (float) -- F1 score on a randomly selected batch from the test set (same one as for loss)
    """
    model.eval()
    with torch.no_grad():
        # Only calculate loss + F1 score on one batch of CV set (increase training speed)
        keeper_idx = random.randint(0, len(loaders["cv"]) - 1)
        for batch_idx, (corrupt_data, target) in enumerate(loaders["cv"]):
            if batch_idx != keeper_idx:
                continue
            pred, mu, logvar = model.forward(corrupt_data)
            loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
            break

    f1 = f1_score(pred, target, replacement_threshold)

    return loss, f1


def auto_garbage_collect(pct=80.0):
    """If memory usage is high, call garbage collector."""
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


def vae_loss(pred, target, mu, logvar):
    """Compute VAE loss for a given batch of genomes to be binary cross entropy
    (BCE) + Kullback-Leibler divergence (KLD)

    Note:
    BCE tries to make the reconstruction as accurate as possible
    KLD tries to make the learned distribtion as similar as possible to the unit Gaussian
    Helpful reading: https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/#what-is-a-variational-autoencoder

    Arguments:
            pred (tensor) -- prediction for which bits should be on in each genome vector
            target (tensor) -- ground truth for which bits should be on in each genome vector
            mu (tensor) -- mean for each latent dimension of the code layer
            logvar (tensor) -- variance for each latent dimension of the code layer

    Returns:
            loss (tensor) -- loss = KLD + BCE
            KLD (tensor) -- KLD loss
            BCE (tensor) -- BCE loss
    """
    # Calculate loss
    # BCE: how well do output + target agree
    BCE = F.binary_cross_entropy(pred, target, reduction="sum")
    # KLD: how much does the learned distribution vary from the unit Gaussian
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = KLD + BCE
    loss = torch.min(loss, 1000000000 * torch.ones_like(loss))  # loss is enormous, clip

    return loss, KLD, BCE


def train_VAE_w_tune(config, reporter):
    """Train autoencoder with ray tune (e.g.: during HP tuning), save model and
    y_probas.

    Arguments:
    config (dict) -- contains parameter and hyperparameter settings for a given trial
            nn_layers -- number of layers in neural net
            weight_decay -- weight_decay
            batch_size - batch_size to use for training data loader
            kfolds -- number of folds for K-fold cross validation
            num_epochs -- number of epochs for which to train
    reporter (progress_reporter) -- ray tune progress reporter
    """
    print(
        "Training model using data files",
        config.TRAIN_DATA_PATH,
        "and",
        config.TEST_DATA_PATH,
    )
    print()

    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cpu")  # "cuda" if use_cuda else "cpu"

    num_features = data.num_features
    model = models.VariationalAutoEncoder(num_features, int(config["nn_layers"]))
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    loaders = cv_dataloader(int(config["batch_size"]), num_features, config["kfolds"])

    for epoch in range(config["num_epochs"]):
        losses = []
        # enumerate batches in epoch
        for batch_idx, (corrupt_data, target) in enumerate(loaders["train"]):
            corrupt_data, target = corrupt_data.to(device), target.to(device)
            optimizer.zero_grad()
            pred, mu, logvar = model.forward(corrupt_data)
            loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # Every 100 batches, take stock of how well things are going
            if (batch_idx + 1) % 100 == 1:
                train_loss = loss.item()
                train_f1 = f1_score(pred, target, config["replacement_threshold"])
                # note that "test_f1 / loss" is actually for a CV fold
                test_loss, test_f1 = cv_vae(
                    model, loaders, config["replacement_threshold"]
                )
                reporter(
                    test_f1=test_f1,
                    train_f1=train_f1,
                    test_loss=test_loss,
                    train_loss=train_loss,
                )
                model.train()

            sys.stdout.flush()
            sys.stderr.flush()

    # save results (will save to tune results dir)
    torch.save(model.state_dict(), "./model.pt")

    # if memory usage is high, may be able to free up space by calling garbage collect
    auto_garbage_collect()


def train_single_vae(
    nn_layers,
    weight_decay,
    lr,
    batch_size,
    kfolds,
    num_epochs,
    replacement_threshold,
    OUT_DIR,
):
    """Train a single VAE (i.e. not during HP tuning), save model and y_probas.

    Arguments:
            nn_layers (int) -- number of layers in neural net
            weight_decay (float) -- weight_decay
            lr (float) -- learning rate
            batch_size (int) - batch_size to use for training data loader
            kfolds (int) -- number of folds for K-fold cross validation
            num_epochs (int) -- number of epochs for which to train
            replacement_threshold (float) -- probability thresh after which to convert bit to 1 vs 0
            OUT_DIR (str) -- path to working dir

    Returns:
            kld (list) -- KLD loss values from training
            bce (list) -- BCE loss values from training
            train_losses (list) -- training loss (BCE + KLD) values from training
            test_losses (list) -- cv loss (BCE + KLD) values from training
            train_f1s (list) -- training F1 values from training
            test_f1s (list) -- cv F1 values from training
            model (genome_embeddings.models.VariationalAutoEncoder) -- trained model
    """
    print(
        "Training model using data files",
        config.TRAIN_DATA_PATH,
        "and",
        config.TEST_DATA_PATH,
    )
    print()

    kld = []
    bce = []
    train_losses = []
    test_losses = []
    train_f1s = []
    test_f1s = []

    device = torch.device("cpu")  # "cuda" if use_cuda else "cpu")

    num_features = data.num_features
    model = models.VariationalAutoEncoder(num_features, nn_layers)
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loaders = cv_dataloader(batch_size, num_features, kfolds)

    for epoch in range(num_epochs):
        # enumerate batches in epoch
        for batch_idx, (corrupt_data, target) in enumerate(loaders["train"]):
            corrupt_data, target = corrupt_data.to(device), target.to(device)
            optimizer.zero_grad()
            pred, mu, logvar = model.forward(corrupt_data)
            loss, KLD, BCE = vae_loss(pred, target, mu, logvar)
            kld.append(KLD)
            bce.append(BCE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # Every 100 batches, take stock of how well things are going
            if batch_idx % 100 == 0:
                train_f1 = f1_score(pred, target, replacement_threshold)
                test_loss, test_f1 = cv_vae(model, loaders, replacement_threshold)
                train_losses.append(loss.item())
                test_losses.append(test_loss.item())
                train_f1s.append(train_f1)
                test_f1s.append(test_f1)
                print("epoch", epoch, "batch", batch_idx)
                print(
                    "train_loss",
                    loss.item(),
                    "train_f1",
                    train_f1,
                    "test_loss",
                    test_loss.item(),
                    "test_f1",
                    test_f1,
                )
                model.train()

            optimizer.step()

            sys.stdout.flush()
            sys.stderr.flush()

    # if memory usage is high, may be able to free up space by calling garbage collect
    auto_garbage_collect()

    return kld, bce, train_losses, test_losses, train_f1s, test_f1s, model


def save_model(name, kld, bce, train_losses, test_losses, train_f1s, test_f1s, model):
    """Save trained model and associated data.

    Arguments:
            name (str) -- path + unique prefix to save files
            kld (list) -- list of KLD losses during training
            bce (list) -- list of BCE losses during training
            train_losses (list) -- training losses (KLD + BCE)
            test_losses (list) -- test losses (KLD + BCE)
            train_f1s (list) -- training F1     scores
            test_f1s (list) -- test F1 scores
            model (genome_embeddings.models.VariationalAutoEncoder) -- trained model
    """
    torch.save(model.state_dict(), name + "_model.pt")
    torch.save(train_losses, name + "_train_losses.pt")
    torch.save(test_losses, name + "_test_losses.pt")
    torch.save(bce, name + "_bce.pt")
    torch.save(kld, name + "_kld.pt")
    torch.save(train_f1s, name + "_train_f1s.pt")
    torch.save(test_f1s, name + "_test_f1s.pt")


def load_model(name):
    """Loads model and associated data from files (training losses, etc)

    Arguments:
            name (str) -- path + common prefix of saved files to load

    Returns:
            model (genome_embeddings.models.VariationalAutoEncoder) -- trained model
            train_losses (list) -- training losses (KLD + BCE)
            test_losses (list) -- test losses (KLD + BCE)
            train_f1s (list) -- training F1     scores
            test_f1s (list) -- test F1 scores
    """
    num_features = data.num_features
    model = models.VariationalAutoEncoder(num_features, 3)
    model.load_state_dict(torch.load(name + "_model.pt"))
    train_losses = torch.load(name + "_train_losses.pt")
    test_losses = torch.load(name + "_test_losses.pt")
    train_f1s = torch.load(name + "_train_f1s.pt")
    test_f1s = torch.load(name + "_test_f1s.pt")

    return model, train_losses, test_losses, train_f1s, test_f1s
