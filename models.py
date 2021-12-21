import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VariationalAutoEncoder(nn.Module):
	"""
	A VariationalAutoEncoder object contains a VAE network comprised of an encoder and decoder. The encoder compresses and reparameterizes input to a constrained multivariate latent distribution. The decoder strives to map a sample from this distribution back to the original version of the input. 
	"""
	def __init__(self, num_features, nn_layers):
		"""
		Construct VAE model
		
		Arguments:
			num_features (int) -- number of features in the dataset
			nn_layers (int) -- number of layers (e.g.: 3 layers = 3 layer encoder, 3 layer decoder)
		"""
		super().__init__()

		self.nn_layers = nn_layers
		self.e_layers = nn.ModuleList()
		self.d_layers = nn.ModuleList()
		self.code_size = 100
		
		# Layers roughly follow log2 reductions -- empirically shown to work well for AEs
		width_custom = {1: [num_features, self.code_size], 
						2: [num_features, 79, self.code_size],
						3: [num_features, 500, 250, self.code_size],
						4: [num_features, 79, 43, 37, self.code_size]}

		width = [num_features]
		
		# Encoder
		in_num = num_features
		for i in range(nn_layers):
			out_num = width_custom[nn_layers][i+1]
			self.e_layers.append(nn.Linear(in_num, out_num))
			nn.init.kaiming_normal_(self.e_layers[-1].weight) # Kaiming / He initialization
			
			old_in_num = in_num # used for second copy of last encoding layer
			in_num = out_num
			width.append(in_num)

		# Add an additional fully connected layer -- need mu and logvar layers
		self.e_layers.append(nn.Linear(old_in_num, self.code_size))
		nn.init.kaiming_normal_(self.e_layers[-1].weight)
	  
		# Decoder
		width = width[::-1]
		width_custom[nn_layers] = width_custom[nn_layers][::-1]
		for i in range(nn_layers):
			out_num = width_custom[nn_layers][i+1]

			# last layer should have exactly num_features features
			# last layer has sigmoid activation, use Xavier instead of He
			if i == (nn_layers - 1): 
				self.d_layers.append(nn.Linear(in_num, num_features))
				nn.init.xavier_normal_(self.d_layers[-1].weight) # Xavier initialization
			else:
				self.d_layers.append(nn.Linear(in_num, out_num))
				nn.init.kaiming_normal_(self.d_layers[-1].weight) # Kaiming / He initialization
			
			in_num = out_num

	def encode(self, x):
		"""
		Map each input data point to mean and logvariance vectors that define the multivariate normal distribution around that data point
		
		Arguments:
			x (tensor) -- [num_features, # of genomes] matrix

		Returns:
			(z_mu, z_logvar) (tensor, tensor) -- ZDMIS = code size; ZDIMS mean and variance units, one for each latent dimension
		"""

		# Encode, apply leaky relu activation to all except last two code layers
		if self.nn_layers > 1:
			for i in range(len(self.e_layers) - 2):
				#print(i)
				x = F.leaky_relu(self.e_layers[i](x))
		
		z_mu = self.e_layers[-2](x)
		z_logvar = self.e_layers[-1](x)  
		# Clip logvar max to prevent infs and Nans downstream
		# Clipping at 100+ will result in infs/nans
		z_logvar = torch.min(z_logvar, 10*torch.ones_like(z_logvar))	
						
		return z_mu, z_logvar

	def reparameterize(self, mu, logvar): 
		"""
		Return a sample from the multivariate normal distribution around each data point
		
		Arguments: 
			mu (tensor) -- mean from the encoder's latent space
			logvar (tensor) -- variance from the encoder's latent space

		Returns:
			sample (tensor) -- sample as if from the input space
		"""
		
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		
		return mu + eps*std
	
	def decode(self, x):
		"""
		Map from latent space samples to reconstruction
		
		Arguments:
			x (tensor) -- sample from the latent space	
		
		Returns:
			out (tensor) -- reconstruction
		"""
		if self.nn_layers > 1:
			for i in range(len(self.d_layers) - 1):				
				x = F.leaky_relu(self.d_layers[i](x))
		
		out = torch.sigmoid(self.d_layers[-1](x))	   
				
		return out
		
	def forward(self, x): 
		"""
		Perform forward pass through VAE
		
		Arguments:
			x (tensor) -- input for VAE
		
		Returns:
			x (tensor) -- generated genomes
			mu (float) -- mean from the encoder's latent space
			logvar (float) -- variance from the encoder's latent space
		"""
		
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		x = self.decode(z)  
		
		return x, mu, logvar