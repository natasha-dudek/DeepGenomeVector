import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VariationalAutoEncoder(nn.Module):
    # useful reading:
    # code example: https://gist.github.com/addy369/9387e4d557cec81ea4848a0dc588a158
    # VAE tutorial (purely theoretical): https://arxiv.org/pdf/1606.05908.pdf
    
    def __init__(self, num_clusters, nn_layers):
        super().__init__()

        # Useful reading:
        # https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12
        # Weight init: https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/        
        
        self.nn_layers = nn_layers
        self.e_layers = nn.ModuleList()
        self.d_layers = nn.ModuleList()
        self.code_size = 100
#        width_custom = {1: [num_clusters, self.code_size], 
#                        2: [num_clusters, 79, self.code_size], 
#                        3: [num_clusters, 79, 43, self.code_size], 
#                        4: [num_clusters, 79, 43, 37, self.code_size]}

        width_custom = {1: [num_clusters, self.code_size], 
                        2: [num_clusters, 79, self.code_size],
                        3: [num_clusters, 500, 250, 100], 
                        #3: [num_clusters, 500, 250, 100],
                        #3: [num_clusters, 1300, 900, 500], 
                        4: [num_clusters, 79, 43, 37, self.code_size]}

        width = [num_clusters]
        
        # Encode
        in_num = num_clusters
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
      
        # Decode
        width = width[::-1]
        width_custom[nn_layers] = width_custom[nn_layers][::-1]
        for i in range(nn_layers):
            out_num = width_custom[nn_layers][i+1]

            # last layer should have exactly num_clusters features
            # last layer has sigmoid activation, use Xavier instead of He
            if i == (nn_layers - 1): 
                self.d_layers.append(nn.Linear(in_num, num_clusters))
                nn.init.xavier_normal_(self.d_layers[-1].weight) # Xavier initialization
            else:
                self.d_layers.append(nn.Linear(in_num, out_num))
                nn.init.kaiming_normal_(self.d_layers[-1].weight) # Kaiming / He initialization
            
            in_num = out_num

    def encode(self, x):
        """
        Arguments:
        x (tensor) -- [num_clusters, # of genomes] matrix

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
        
        if self.nn_layers > 1:
            for i in range(len(self.d_layers) - 1):                
                x = F.leaky_relu(self.d_layers[i](x))
                       
        return torch.sigmoid(self.d_layers[-1](x))
        
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

class AutoEncoder(nn.Module):

    def __init__(self, num_clusters, nn_layers):
        super(AutoEncoder, self).__init__()

        # Useful reading:
        # https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12
        # Weight init: https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/
        self.nn_layers = nn_layers
        self.layers = nn.ModuleList()
        
        #width_custom = {1: [7065, 13], 2: [7065, 79, 13], 3: [7065, 79, 43, 13], 4: [7065, 79, 43, 37, 13]}
        #width_custom = {1: [9874, 13], 2: [9874, 79, 13], 3: [9874, 79, 43, 13], 4: [9874, 79, 43, 37, 13]}
        width_custom = {1: [9874, 13], 2: [9874, 150, 100], 3: [9874, 500, 250, 100], 4: [9874, 79, 43, 37, 13]}


        width = [num_clusters]
        # Encoder
        in_num = num_clusters
        for i in range(nn_layers):

            #out_num = math.ceil(math.log2(in_num))
            #if out_num < 13: out_num = 13
            out_num = width_custom[nn_layers][i+1]             

            self.layers.append(nn.Linear(in_num, out_num))
            nn.init.kaiming_normal_(self.layers[-1].weight) # Kaiming / He initialization
            in_num = out_num
            width.append(in_num)

        # Decoder
        width = width[::-1]
        width_custom[nn_layers] = width_custom[nn_layers][::-1]
        for i in range(nn_layers):
            #out_num = width[i+1]
            out_num = width_custom[nn_layers][i+1]

            # last layer should have exactly num_clusters features
            # last layer has sigmoid activation, use Xavier instead of He
            if i == (nn_layers - 1): 
                self.layers.append(nn.Linear(in_num, num_clusters))
                nn.init.xavier_normal_(self.layers[-1].weight)
            else:
                self.layers.append(nn.Linear(in_num, out_num))
                nn.init.kaiming_normal_(self.layers[-1].weight) # Kaiming / He initialization
            in_num = out_num

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