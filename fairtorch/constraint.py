import torch
from torch import nn
from torch.nn import functional as F


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, alpha=1, norm=2):
        super(ConstraintLoss, self).__init__()
        self.alpha = alpha
        self.norm = norm
        self.n_class = n_class
        self.n_constr = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constr, self.dim_condition))
        self.c = torch.zeros(self.n_constr)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constr)

    def forward(self, X, out, sensitive, y=None):
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(torch.mv(self.M, mu) - self.c)
        if self.norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons


class DemographicParityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(DemographicParityLoss, self).__init__(n_class=self.n_class, alpha=alpha, norm=norm)
        self.n_constr = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constr, self.dim_condition))
        for i in range(self.n_constr):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constr)

    def mu_f(self, X, out, sensitive, y=None):
        list_Es = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v  # torch.bool
            list_Es.append(out[idx_true].mean())
        list_Es.append(out.mean())
        return torch.stack(list_Es)

    def forward(self, X, out, sensitive):
        return super(DemographicParityLoss, self).forward(X, out, sensitive)
