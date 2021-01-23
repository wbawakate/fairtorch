import torch
from torch import nn
from torch.nn import functional as F




class DiscriminatorNet(nn.Module):

    def __init__(self, **kargs):
        """
        Discriminator of adversarial traing
        
        Parameters
        -----------------
        n_layers: int, default 1
            number of layers
        dim_input: int, default 1
            dimention of input
        dim_hidden: int, default 16
            dimention of hidden layers
        """
        super(DiscriminatorNet, self).__init__()
        self.n_layers = kargs["n_layers"]
        self.dim_input = kargs["dim_input"]
        self.dim_hidden = kargs["dim_hidden"]
        self.nets = nn.ModuleList([])
        if self.n_layers == 1:
            self.nets.append(nn.Linear(self.dim_input, 1) )
        else:
            for i in range(1, self.n_layers + 1):
                if i == self.n_layers:
                    dim_out = 1
                else:
                    dim_out = self.dim_hidden
                if i == 1:
                    dim_in = self.dim_input
                else:
                    dim_in = self.dim_hidden
                self.nets.append(nn.Linear(dim_in, dim_out))
                if i != self.n_layers:
                    self.nets.append(nn.ReLU())
            
    
    def forward(self, x):
        for i in range(len(self.nets)):
            x = self.nets.forward(x)
        return x

        


class AdversarialDebiasingLoss(nn.Module):

    def __init__(self, **kargs):
        super(AdversarialDebiasingLoss, self).__init__()

    def forward(self, X, out, sensitive, y=None):
        return 0