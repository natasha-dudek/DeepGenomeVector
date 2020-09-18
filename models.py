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
    # great VAE tutorial (purely theoretical): https://arxiv.org/pdf/1606.05908.pdf
    
    def __init__(self, num_clusters, nn_layers):
        super().__init__()

        # Useful reading:
        # https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124
        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12
        # Weight init: https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/
#
#        self.code_size = 13
#        self.nn_layers = nn_layers
        
#        self.enc1 = nn.Linear(in_features=num_clusters, out_features=79)
#        self.mu = nn.Linear(in_features=79, out_features=self.code_size)
#        self.logvar = nn.Linear(in_features=79, out_features=self.code_size)
#        
#        self.dec1 = nn.Linear(in_features=self.code_size, out_features=79)
#        self.dec2 = nn.Linear(in_features=79, out_features=num_clusters)
        
        
        self.nn_layers = nn_layers
        self.e_layers = nn.ModuleList()
        self.d_layers = nn.ModuleList()
        self.code_size = 13
        width_custom = {1: [num_clusters, self.code_size], 
                        2: [num_clusters, 79, self.code_size], 
                        3: [num_clusters, 79, 43, self.code_size], 
                        4: [num_clusters, 79, 43, 37, self.code_size]}
        width = [num_clusters]
        
        # Encode
#        in_num = num_clusters
#        for i in range(nn_layers):
#            out_num = width_custom[nn_layers][i+1]
#
#            self.e_layers.append(nn.Linear(in_num, out_num))
#            nn.init.kaiming_normal_(self.e_layers[-1].weight) # Kaiming / He initialization
#            
#            old_in_num = in_num # used for second copy of last encoding layer
#            in_num = out_num
#            width.append(in_num)

        # Now add an additional fully connected layer 
        #self.e_layers.append(nn.Linear(old_in_num, self.code_size))
        #nn.init.kaiming_normal_(self.e_layers[-1].weight)
#        self.e_layers.append(nn.Linear(old_in_num, self.code_size))
#        nn.init.kaiming_normal_(self.e_layers[-1].weight)
        
        in_num = num_clusters
        width = [num_clusters, 79, 13] 
        self.enc1 = nn.Linear(num_clusters, 79)
        self.enc21 = nn.Linear(79, self.code_size)
        self.enc22 = nn.Linear(79, self.code_size)
        
        self.dec1 = nn.Linear(self.code_size, 79)
        self.dec2 = nn.Linear(79, num_clusters)
        
        # Decode
#        width = width[::-1]
#        width_custom[nn_layers] = width_custom[nn_layers][::-1]
#        for i in range(nn_layers):
#            out_num = width_custom[nn_layers][i+1]
#
#            # last layer should have exactly num_clusters features
#            # last layer has sigmoid activation, use Xavier instead of He
#            if i == (nn_layers - 1): 
#                self.d_layers.append(nn.Linear(in_num, num_clusters))
#                nn.init.xavier_normal_(self.d_layers[-1].weight) # Xavier initialization
#            else:
#                self.d_layers.append(nn.Linear(in_num, out_num))
#                nn.init.kaiming_normal_(self.d_layers[-1].weight) # Kaiming / He initialization
#            
#            in_num = out_num
#
    def encode(self, x):
        """
        Arguments:
        x (tensor) -- [num_clusters, # of genomes] matrix

        Returns:
        (z_mu, z_logvar) (tensor, tensor) -- ZDMIS = code size; ZDIMS mean and variance units, one for each latent dimension
        """
        if torch.isnan(x.max()):
            print('X going in encode is Nan') 
        # Encode, apply leaky relu activation to all except last two code layers
        #x = self.e_layers[0](x)
        x = self.enc1(x)
        if torch.isnan(x.max()):
            print('X after layer 0 is Nan, params is:', torch.max(torch.nn.utils.parameters_to_vector(self.enc1.parameters())))

        x = F.leaky_relu(x)
        
        if torch.isnan(x.max()):
            print('X after relu is Nan') 
        
        z_mu = self.enc21(x)
        #z_mu = self.e_layers[1](x)
        
        if torch.isnan(z_mu.max()):
            print('z_mu is Nan') 
         
        z_logvar = self.enc22(x)   
        #z_logvar = self.e_layers[2](x) 
               
        if torch.isnan(z_logvar.max()):
            print('z_logvar is Nan') 
            
                    
#        for i in range(len(self.e_layers) - 2):
#            if torch.isnan(x.max()):
#                print("coming in bad",i)
#                
#            x = self.e_layers[i](x)
#            
#            if torch.isnan(x.max()):
#                print("through layer",i)
#                
#            x = F.leaky_relu(x)
#            
#            if torch.isnan(x.max()):
#                print("post-relu",i)
#                
#        if torch.isnan(x.max()):
#            print("vhfioahgvrueihgq",i)
        
#        z_mu = self.e_layers[-2](x)
#        z_logvar = self.e_layers[-1](x)
        z_mu = torch.min(z_mu, 100*torch.ones_like(z_mu))
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

#        mu = torch.min(mu, 100*torch.ones_like(mu))
#        logvar = torch.min(logvar, 10*torch.ones_like(logvar))    
                
        std = torch.exp(0.5*logvar)
        
        std = torch.min(std, 1000*torch.ones_like(std)) 
        
        print("mu.max()", mu.max(), "logvar.max()", logvar.max(), "std.max()", std.max())
        
        if torch.isnan(std.max()):
            print("std is NaN")
        
        eps = torch.randn_like(std)
        
        return mu + eps*std
    
    def decode(self, x):
        
#        for i in range(len(self.d_layers) - 1):
#            x = F.leaky_relu(self.d_layers[i](x))
#                       
#        return torch.sigmoid(self.d_layers[-1](x))

        x = F.relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x

        
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
        if torch.isnan(x.max()):
            print('X orig is Nan') 
        
        mu, logvar = self.encode(x)
        
        if torch.isnan(mu.max()):
            print('mu encode layer is Nan')
        if torch.isnan(logvar.max()):
            print('logvar encode layer is Nan')
            
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        
        if torch.isnan(x.max()):
            print('X final is Nan')         
        
        return x, mu, logvar

        