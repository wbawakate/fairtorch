import torch
from torch import nn
from torch.nn import functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = nn.Parameter(torch.zeros((self.n_constraints, self.dim_condition)) )# 4x3
        self.c = nn.Parameter(torch.zeros(self.n_constraints))
        self.M.requires_grad = False
        self.c.requires_grad = False


    def mu_f(self, X=None, out=None, sensitive=None, y=None):
        ze = torch.randn(self.dim_condition).to(X.device)
        return ze * torch.max(X) *  torch.mean(out)

    def forward(self, X, out, sensitive, y=None):
        sensitive1 = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.reshape(out.shape)
        out1 = torch.sigmoid(out)
        mu1 = self.mu_f(X=X, out=out1, sensitive=sensitive1, y=y)
        print(mu1, mu1.device)
        gap_constraint = F.relu(torch.matmul(self.M, mu1) - self.c)
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons


class DemographicParityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(DemographicParityLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                M[i, j - 1] = -1.0
                M[i, -1] = 1.0
        c = torch.zeros(self.n_constraints)
        self.M = nn.Parameter( M)# 4x3
        self.c = nn.Parameter(c)
        self.M.requires_grad = False
        self.c.requires_grad = False


    def mu_f(self, X, out, sensitive, y=None):
        expected_values_list = [] # ここか？　listをやめてみよう
        for v in self.sensitive_classes:
            idx_true = sensitive == v  # torch.bool
            expected_values_list.append(torch.mean(out[idx_true]))
        expected_values_list.append(torch.mean(out))
        return torch.stack(expected_values_list, dim=0)

    def forward(self, X, out, sensitive, y=None):
        return super(DemographicParityLoss, self).forward(X, out, sensitive)


class EqualiedOddsLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.y_classes = [0, 1]
        self.n_class = len(sensitive_classes)
        self.n_y_class = len(self.y_classes)
        super(EqualiedOddsLoss, self).__init__(n_class=self.n_class, alpha=alpha, p_norm=p_norm)
        # K:  number of constraint : (|A| x |Y| x {+, -})
        self.n_constraints = self.n_class * self.n_y_class * 2
        # J : dim of conditions  : ((|A|+1) x |Y|)
        self.dim_condition = self.n_y_class * (self.n_class + 1)
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        # make M (K * J): (|A| x |Y| x {+, -})  *   (|A|+1) x |Y|) )
        self.c = torch.zeros(self.n_constraints)
        element_K_A = self.sensitive_classes + [None]
        for i_a, a_0 in enumerate(self.sensitive_classes):
            for i_y, y_0 in enumerate(self.y_classes):
                for i_s, s in enumerate([-1, 1]):
                    for j_y, y_1 in enumerate(self.y_classes):
                        for j_a, a_1 in enumerate(element_K_A):
                            i = i_a * (2 * self.n_y_class) + i_y * 2 + i_s
                            j = j_y + self.n_y_class * j_a
                            self.M[i, j] = self.__element_M(a_0, a_1, y_1, y_1, s)

    def __element_M(self, a0, a1, y0, y1, s):
        if a0 is None or a1 is None:
            x = y0 == y1
            return -1 * s * x
        else:
            x = (a0 == a1) & (y0 == y1)
            return s * float(x)

    def mu_f(self, X, out, sensitive, y):
        expected_values_list = []
        for u in self.sensitive_classes:
            for v in self.y_classes:
                idx_true = (y == v) * (sensitive == u)  # torch.bool
                expected_values_list.append(out[idx_true].mean())
        # sensitive is star
        for v in self.y_classes:
            idx_true = y == v
            expected_values_list.append(out[idx_true].mean())
        return torch.cat(expected_values_list, dim=0)

    def forward(self, X, out, sensitive, y):
        return super(EqualiedOddsLoss, self).forward(X, out, sensitive, y=y)

def genelate_data(n_samples = 1000, n_feature=5):
    
    y = np.random.randint(0, 2, size=n_samples)
    loc0 = np.random.uniform(-2, 2, n_feature)
    loc1 = np.random.uniform(-2, 2, n_feature)

    X = np.zeros((n_samples, n_feature))
    for i, u in enumerate(y):
        if y[i] ==0:
            X[i] = np.random.normal(loc = loc0, scale=1.0, size=n_feature)  
        else:
            X[i] = np.random.normal(loc = loc1, scale=1.0, size=n_feature)  

    sensi_feat = (X[:, 0] > X[:, 0].mean()).astype(int)
    X[:, 0] = sensi_feat.astype(np.float32)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    sensi_feat = torch.from_numpy(sensi_feat)
    return X, y, sensi_feat

def main():
    n_samples = 512
    n_feature = 5

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    dataset = genelate_data(1024, n_feature=n_feature)
    # data split
    n_train = int(0.7*len(dataset[0]))
    X_train, y_train, sensi_train = map(lambda x : x[:n_train], dataset)
    X_test, y_test, sensi_test = map(lambda x : x[n_train:], dataset)

    assert len(X_train) == len(X_train)
    assert len(X_test) == len(X_test)

    model = nn.Sequential(nn.Linear(n_feature,1))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    model.train()
    for i in range(0, 20):
        optimizer.zero_grad()    
        X_train = X_train.to(device)
        logit = model(X_train)
        y_train = y_train.to(device)
        loss = criterion(logit.view(-1), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        y_pred = (F.sigmoid(model(X_test.to(device))).reshape(-1) > 0.5 ).cpu().float()
        acc_test = (y_pred  == y_test ).float().mean().item()

    print("acc test: ",acc_test)

    acc_test_vanilla = acc_test

    gap_vanilla = np.abs(y_pred[sensi_test==0].mean().item() - y_pred[sensi_test==1].mean().item())
    print("gap of expected values: ", gap_vanilla)
    

    dim_hiden = 32
    model = nn.Sequential(nn.Linear(n_feature,1))
    model.to(device)
    cons_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=100, p_norm=2) # constraint 
    # cons_loss = ConstraintLoss(n_class=2, alpha=100, p_norm=2) # constraint 
    cons_loss.to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.1)

    print("cons_loss.M.device= ", cons_loss.M.device)
    model.train()
    # train 
    for i in range(0, 10):
        optimizer.zero_grad()    
        X_train = X_train.to(device)
        logit = model(X_train)
        y_train = y_train.to(device)
        sensi_train = sensi_train.to(device)
        loss = criterion(logit.reshape(-1), y_train)
        loss +=  cons_loss(X_train, logit, sensi_train) # add constraint
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        y_pred = (F.sigmoid(model(X_test.to(device))).reshape(-1) > 0.5 ).cpu().float()
        acc_test = (y_pred  == y_test ).float().mean().item()

    print("acc test: ",acc_test)

    acc_test_vanilla = acc_test

    gap_dp = np.abs(y_pred[sensi_test==0].mean().item() - y_pred[sensi_test==1].mean().item())
    print("gap of expected values: ", gap_dp)



if __name__ == '__main__':
    main()
        