import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

import numpy as np

class CNN_Encoder(nn.Module):
    """ Encodes input data to a latent space representation """
    def __init__(self, latent_dim, input_size):
        super(CNN_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.channel_mult = 8

        # Define the convolutional layers sequence, use kernel size: 3, stride: 2
        self.conv = nn.Sequential(
            nn.Conv2d(1,self.channel_mult*1, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult),
            nn.ReLU(),
            
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(),
            
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(),
    
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(1280, self.latent_dim, bias=True), # 1280: output size of the last CNN layer; 
        )

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        return x

class CNN_Decoder(nn.Module):
    """ Decodes latent space representation back into the input data format. """
    def __init__(self, latent_dim, input_size):
        super(CNN_Decoder, self).__init__()
        self.channel_mult = 8
        self.output_channels = 1
        self.latent_dim = latent_dim

        self.latent2dec = nn.Sequential(
            nn.Linear(self.latent_dim,1280,bias=True),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(     
            nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*4, 3, 2, (0,1)),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 3, 2, (1,0)),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(self.channel_mult, 1, 3, 2, 1),
        )

    def forward(self, x):
        x = self.latent2dec(x)
        x = x.view(-1, 64, 5, 4) # reshape
        x = self.deconv(x)[:,:,2:-3,:-1] # crop to the same size as input
        return x
        
class autoencoder(nn.Module):
    """
    Autoencoder includes encoder and decoder for pretrain
    """
    def __init__(self, args):
        super(autoencoder, self).__init__()
        self.input_size = args.input_size
        self.latent_dim = args.latent_dim
        self.encoder = CNN_Encoder(self.latent_dim, self.input_size)
        self.decoder = CNN_Decoder(self.latent_dim, self.input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, *self.input_size))
        x = self.decode(z)
        return x,z
    
class clustering(nn.Module):
    """
    Clustering layer connected to latent representation
    input: latent vectors (n_samples, latent_dim)
    output: soft labels for each sample (n_samples, n_clusters)
    """
    def __init__(
        self, 
        n_clusters:int,
        input_shape:int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None
        ) -> None:
        super(clustering, self).__init__()

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_shape = input_shape

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_clusters, self.input_shape, dtype=torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.clustcenters = nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        """ student t-distribution, or soft label. """
        q = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(inputs, axis=1) - self.clustcenters, 2), axis=2) / self.alpha))
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
class DEC_net(nn.Module):
    """
    Deep Embedded Clustering (DEC) Network.
    Combines the encoder, decoder, and clustering layer for end-to-end training.
    """
    def __init__(self, args):
        super(DEC_net, self).__init__()
        
        self.n_clusters = args.n_clusters
        self.input_size = args.input_size
        input_shape = args.latent_dim
        self.AE = autoencoder(args)
        # self.AE.load_state_dict(torch.load(args.init_ae_model, map_location=args.device)) # Load pre-trained AE
        self.clusterlayer = clustering(self.n_clusters, input_shape)   
        
    def forward(self, x):
        z = self.AE.encode(x.view(-1, *self.input_size)) # Latent representation
        x = self.AE.decode(z) # Reconstruction
        q = self.clusterlayer(z) # Clustering output
        return q, x, z