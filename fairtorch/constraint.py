import numpy as np
from torch.nn import functional as F
from torch import nn
import torch

class ConstrLoss(nn.Module):
    def __init__(self, n_class=2, B=1, norm=2):
        super(ConstrLoss, self).__init__()
        self.B = B
        self.norm = norm
        self.n_class = n_class
        self.K = 2 
        self.J = self.n_class + 1 
        self.M = torch.zeros((self.K, self.J))
        self.c = torch.zeros(self.K)
    
    def mu_f(self, X=None, y=None, A=None):
        return torch.zeros(self.K)

    def forward(self, X, out, A, y=None):
        mu = self.mu_f(X=X, out=out, A=A, y=y)
        b = F.relu(torch.mv(self.M, mu) -  self.c)
        if self.norm == 2:
            cons = self.B * torch.dot(b, b)
        else:
            cons = self.B * torch.dot(b.detach(), b) 
        return cons


class DPLoss(ConstrLoss):
    def __init__(self,  A_classes=[0, 1], B=1, norm=2):
        """loss of demograpfhic parity

        Args:
            A_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            B (int, optional): [description]. Defaults to 1.
            norm (int, optional): [description]. Defaults to 2.
        """        
        self.A_classes = A_classes
        self.n_class = len(A_classes)
        super(DPLoss, self).__init__(n_class=self.n_class, B=B, norm=norm)
        self.K = 2* self.n_class
        self.J = self.n_class + 1
        self.M = torch.zeros((self.K, self.J))
        for i in range(self.K):
            j = i % 2
            if j==0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j-1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.K)
        
    def __mu_f(self, X, out, y, A): # 旧バージョン
        # ここから怪しい
        if isinstance(A, torch.Tensor):
            uniques = torch.unique(A.detach().cpu())
        elif isinstance(A, list):
            uniques = torch.Tensor(list(set(A)))
        elif isinstance(A, np.array):
            uniques = torch.Tensor(np.unique())
        else:
            raise TypeError("A is expected to be a torch.Tesor, "  
                 "list or numpy, but passed {}".format(type(A)))
        list_Es = []
        for i in range(uniques.size(0)):
            idx_true = A == uniques[i] # torch.bool
            list_Es.append(out[idx_true].mean())
        list_Es.append(out.mean())
        return torch.stack(list_Es)

    def mu_f(self, X, out, A, y=None):
        list_Es = []
        for v in self.A_classes:
            idx_true = A == v # torch.bool
            list_Es.append(out[idx_true].mean())
        list_Es.append(out.mean())
        return torch.stack(list_Es)

    def forward(self, X, out, A):
        return super(DPLoss, self).forward(X, out, A)

        
