import torch
from torch import nn
from torch.nn import functional as F


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, norm_dimension=2):
        super(ConstraintLoss, self).__init__()
        self.norm_dimension = norm_dimension
        self.n_class = n_class
        self.constraints_dim = 2
        self.J = self.n_class + 1
        self.M = torch.zeros((self.constraints_dim, self.J))
        self.constraints_value = torch.zeros(self.constraints_dim)

    def mu_f(self, X=None, y=None, A=None):
        return torch.zeros(self.K)

    def forward(self, X, out, A, y=None):
        mu = self.mu_f(X=X, out=out, A=A, y=y)
        constraints_resudual = F.relu(torch.mv(self.M, mu) - self.constraints_value)
        if self.norm_dimension == 2:
            cons = self.B * torch.dot(constraints_resudual, constraints_resudual)
        else:
            cons = self.B * torch.dot(constraints_resudual.detach(), constraints_resudual)
        return cons


class DemographicParityLoss(ConstraintLoss):
    def __init__(self, A_classes=[0, 1], B=1, norm_dimension=2):
        """loss of demograpfhic parity

        Args:
            A_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            B (int, optional): [description]. Defaults to 1.
            norm_dimension (int, optional): [description]. Defaults to 2.
        """
        self.A_classes = A_classes
        self.n_class = len(A_classes)
        super(DemographicParityLoss, self).__init__(
            n_class=self.n_class, B=B, norm_dimension=norm_dimension
        )
        self.constraints_dim = 2 * self.n_class
        self.J = self.n_class + 1
        self.M = torch.zeros((self.constraints_dim, self.J))
        for i in range(self.K):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.constraints_value = torch.zeros(self.constraints_dim)

    def mu_f(self, X, out, A, y=None):
        list_Es = []
        for v in self.A_classes:
            idx_true = A == v  # torch.bool
            list_Es.append(out[idx_true].mean())
        list_Es.append(out.mean())
        return torch.stack(list_Es)

    def forward(self, X, out, A):
        return super(DemographicParityLoss, self).forward(X, out, A)
