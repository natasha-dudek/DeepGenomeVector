import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
	
	def __init__(self, num_clusters):
		super(AutoEncoder, self).__init__()
		
		n = 1
		
#		for i in len(nn_layers/2):
#			foo = nn.Linear(num_clusters, int(num_clusters/(2*i)))
#			setattr(self, 'e1', foo)
		
				
		# Encoder
		self.e1 = nn.Linear(num_clusters, int(num_clusters/(2*n)))
		self.e2 = nn.Linear(int(num_clusters/(2*n)), int(num_clusters/(4*n)))
		self.e3 = nn.Linear(int(num_clusters/(4*n)), int(num_clusters/(6*n)))
		self.e4 = nn.Linear(int(num_clusters/(6*n)), int(num_clusters/(8*n)))
		self.e5 = nn.Linear(int(num_clusters/(8*n)), int(num_clusters/(10*n)))
		
		# Latent View (bottleneck layer)
		self.lv = nn.Linear(int(num_clusters/(10*n)), int(num_clusters/(12*n)))
		
		# Decoder
		self.d1 = nn.Linear(int(num_clusters/(12*n)), int(num_clusters/(10*n)))
		self.d2 = nn.Linear(int(num_clusters/(10*n)), int(num_clusters/(8*n)))
		self.d3 = nn.Linear(int(num_clusters/(8*n)), int(num_clusters/(6*n)))
		self.d4 = nn.Linear(int(num_clusters/(6*n)), int(num_clusters/(4*n)))
		self.d5 = nn.Linear(int(num_clusters/(4*n)), int(num_clusters/(2*n)))
		self.output_layer = nn.Linear(int(num_clusters/(2*n)), num_clusters)
		
	def forward(self,x):
		
		#getattr
		
		x = F.relu(self.e1(x))
		x = F.relu(self.e2(x))
		x = F.relu(self.e3(x))
		x = F.relu(self.e4(x))
		x = F.relu(self.e5(x))
		
		x = torch.sigmoid(self.lv(x))
		
		x = F.relu(self.d1(x))
		x = F.relu(self.d2(x))
		x = F.relu(self.d3(x))
		x = F.relu(self.d4(x))
		x = F.relu(self.d5(x))
		
		x = torch.sigmoid(self.output_layer(x)) 
		
		return x

	
	def encode(self, x):
		# Return autoencoder latent space
		x = F.relu(self.e1(x))
		x = F.relu(self.e2(x))
		x = F.relu(self.e3(x))
		x = F.relu(self.e4(x))
		x = F.relu(self.e5(x))
				
		#x = torch.sigmoid(self.lv(x))
		
		return x