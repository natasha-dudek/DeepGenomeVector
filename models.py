import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
	
	def __init__(self, num_clusters, nn_layers):
		super(AutoEncoder, self).__init__()
		
		# Useful reading:
		# https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124
		# https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12
		# Weight init: https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/
		self.nn_layers = nn_layers
		self.layers = nn.ModuleList()

		# Encoder
		division = 2
		for i in range(int(nn_layers/2)):
			if i == 0: in_num = 1
			else: in_num = division-2
			self.layers.append(nn.Linear(int(num_clusters/in_num), int(num_clusters/division)))
			nn.init.kaiming_normal_(self.layers[-1].weight) # Kaiming / He initialization
			division += 2
		
		# Decoder
		division -= 2
		for i in range(int(nn_layers/2)):
			if i == int(nn_layers/2) - 1: out = 1
			else: out = division-2			
			self.layers.append(nn.Linear(int(num_clusters/division), int(num_clusters/out)))
						
			if i == (int(nn_layers/2) - 1): # last layer has sigmoid activation, use Xavier instead of He
				nn.init.xavier_normal_(self.layers[-1].weight)
			else:
				nn.init.kaiming_normal_(self.layers[-1].weight) # Kaiming / He initialization
				
			division -= 2

		
	def forward(self, x): 
		
		# apply leaky relu to all except last layer of nn
		for i in range(len(self.layers) - 1):
			x = F.leaky_relu(self.layers[i](x))
		# Apply sigmoid to final output layer of decoder 
		x = torch.sigmoid(self.layers[len(self.layers) - 1](x))
		
		return x

	
	def encode(self, x):

		# Encode, apply leaky relu activation to all except code layer
		for i in range(int(len(self.layers)/2) - 1):
			x = F.leaky_relu(self.layers[i](x))
		# Do not add non-linearity to code layer
		x = self.layers[int(len(self.layers)/2) - 1](x)
					
		return x