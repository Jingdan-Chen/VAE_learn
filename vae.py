import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, hidden_dim_2=200, latent_dim=2):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim_2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim_2, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
    
def loss_function(x, x_hat, mean, log_var, alpha=.5):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return 2 * ((1-alpha) * reproduction_loss + alpha * KLD)

class CVAE(nn.Module):
    def __init__(self, input_dim_x=784, input_dim_y=10, hidden_dim=400, hidden_dim_2=200, latent_dim=2):
        super(CVAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_x + input_dim_y, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim_2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim_2, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim_y, hidden_dim_2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim_x),
            nn.Sigmoid()
            )
     
    def encode(self, x, y):
        input_ = torch.cat((x, y), dim=1)
        x = self.encoder(input_)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, z, y):
        input_ = torch.cat((z, y), dim=1)
        return self.decoder(input_)

    def forward(self, x, y):
        mean, log_var = self.encode(x, y)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z, y)
        return x_hat, mean, log_var