import torch
import torch.nn as nn
import sklearn as sk
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score, pairwise_distances
from torch.autograd import Variable
import random
import numpy as np

def f1_score(pred_non_bin, target, replacement_threshold):
	"""
	Calculates F1 score
	
	Arguments:
	pred_non_bin -- torch tensor containing predictions output by model (probability values)
	dataset -- torch tensor containing true values (binary)
	
	Returns:
	avg_f1 -- average F1 score
	"""
	binarized_preds = binarize(pred_non_bin, replacement_threshold)
	
	f1s = []
	for i in range(0,len(binarized_preds)):
		f1 = sk.metrics.f1_score(target.data.numpy()[i], binarized_preds[i])
		f1s.append(f1)
	
	return sum(f1s)/len(f1s)
	
def hamming_dist(pred_non_bin, target, replacement_threshold):
	"""
	Calculates hamming distance
		
	Arguments:
	pred_non_bin -- torch tensor containing predictions output by model (probability values)
	target -- torch tensor containing true values (binary)
	
	Returns:
	avg_hamm -- avg hamming distance between predictions and targets in the dataset
	"""
	binarized_preds = binarize(pred_non_bin, replacement_threshold)
	
	hamms = []
	for i in range(0,len(binarized_preds)):
		#hamm = sk.metrics.f1_score(target.data.numpy()[i], binarized_preds[i])
		hamm = pairwise_distances(target.data.numpy()[i], binarized_preds[i], metric='hamming')
		hamms.append(hamm)
	
	return sum(hamms)/len(hamms)

def binarize(pred_tensor, replacement_threshold):
	"""
	Values below or equal to threshold are replaced by 0, else by 1
	
	Arguments:
	pred_tensor -- numpy array from pred.detach().numpy()
	formerly: torch tensor of probabilities ranging from 0 to 1
	threshold -- threshold at which to replace with 0 vs 1
	
	Returns:
	binary_preds -- list of numpy array of 0's and 1's 
	"""
		
	binary_preds = []
	for i in pred_tensor:
		try:
			pred_arr = i.detach().numpy() #.data.numpy()		
		except AttributeError:
			pred_arr = i
		b = Binarizer(threshold=replacement_threshold).fit_transform(pred_arr.reshape(1, -1))
		binary_preds.extend(b)
	return binary_preds

def generate_embeddings(model, test_data):
	with torch.no_grad():
		model.eval()
		embeddings = model.encode(test_data)
	return embeddings

def pick_loader(epoch, stage, loaders):
	"""
	When using cirriculum learning, selects right dataloader to use based on where you are in the cirriculum
	
	Arguments:
	epoch -- which epoch are you on in training
	stage -- length(train_data) / 3 (3 part of cirriculum)
	loaders -- dict of list of loaders
		loaders['train'][0] = small sized genomes
		loaders['train'][1] = medium sized genomes
		loaders['train'][2] = large sized genomes
	
	Returns:
	loader -- dataloader (small, medium, or large, based on where you are in training)
	"""
		
	if epoch < stage: # use small genes dataloader
		loader = loaders['train'][0]
		load_tracker = "small"
	elif 2*stage > epoch >= stage: # use medium genes dataloader
		loader = loaders['train'][1]
		load_tracker = "medium"
	else: # use large genes dataloader
		loader = loaders['train'][2]
		load_tracker = "large"
	
	return loader, load_tracker
	
def train_model(loaders, model, lr, num_epochs, print_every, batch_size, SAVE_FP, replacement_threshold, cluster_names, cirriculum, weight_decay, train_data):
	# define optimization strategy
	# L2 regularization in pytorch: https://discuss.pytorch.org/t/simple-l2-regularization/139
	# Adam weight decay: https://www.fast.ai/2018/07/02/adam-weight-decay/
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # weight_decay term = L2 regularization
	# weight loss function to counteract data imbalance
#	total = train_data.shape[0]
#	pos = np.sum(train_data.detach().numpy(), axis=0)
#	neg = total - pos
	#pos_weight = torch.tensor(neg/pos)
	#criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
	#criterion = nn.BCEWithLogitsLoss(reduction='sum')
	criterion = nn.BCELoss(reduction='sum')
	#criterion = nn.BCEWithLogitsLoss(reduction='sum')
	
	# Use gpu if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device, dtype= torch.float)
	# Specify train mode
	model.train()
	# track loss
	train_losses = []
	test_losses = []
	# track F1
	train_f1_scores = []
	test_f1_scores = []
	
	# For cirriculum learning: divide epochs into three stages
	if cirriculum: stage = int(num_epochs/3)
	
	# enumerate epochs
	for epoch in range(num_epochs):
		
		if cirriculum:
			loader, load_tracker = pick_loader(epoch, stage, loaders)
		else:
			loader = loaders['train']
			load_tracker = "single" # will print this in progress update
			
		# enumerate mini-batches
		for idx, (train_data, target) in enumerate(loader):
			
			# Ensure train mode is on
			model.train()
			
			# Zero gradients
			optimizer.zero_grad()
			# Compute the model output
			pred = model(train_data)
			# Calculate and save loss
			loss = criterion(pred, target)
			# Perform backpropagation
			loss.backward()
			# Update weights
			optimizer.step()
			
			train_f1 = f1_score(pred, target, replacement_threshold)
			
			if idx % print_every == 1:
				
				# Get training loss, F1 score
				#train_losses.append(loss.cpu().data.item()) 
				train_losses.append(loss.item()) 
				train_f1 = f1_score(pred, target, replacement_threshold)
				train_f1_scores.append(train_f1)

				# Print progress update to display
				print('\r Training epoch {}/{}, batch {}/{}, {} loader, \tLoss: {:.2f}'.format(
					epoch + 1, # current epoch
					num_epochs, # total number of epochs
					idx ,#+ 1, # batch 
					len(loaders['train']), # total number data pts per epoch
					load_tracker, # which loader are you using (cirriculum learning)
					loss.cpu().data.item() # loss
					)
					) 
				
				# Get test loss, F1 score
				model.eval()
				with torch.no_grad():
					# Only keep one loss + F1 score
					# Need to have equal number of train and test cases for learning curve
					keeper_idx = random.randint(0,len(loaders['test'])-1)
					for test_idx, (test_data,target) in enumerate(loaders['test']):  
						if test_idx != keeper_idx: continue
						pred = model(test_data)
						loss = criterion(pred, target)
						test_losses.append(loss.item()) 
						test_f1 = f1_score(pred, target, replacement_threshold)
						test_f1_scores.append(test_f1)
						break
	
	#torch.save(model.state_dict(), SAVE_FP+"autoencoder_weights.txt")
	torch.save(model, SAVE_FP+"autoencoder_weights.txt")

	return train_losses, test_losses, train_f1_scores, test_f1_scores
