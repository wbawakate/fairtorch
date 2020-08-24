import os
import random
from random import randint

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from fairtorch import ConstraintLoss, DemographicParityLoss


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(2020)


class SensitiveDataset(Dataset):
    def __init__(self, x, y, sensitive):
        self.x = x
        self.y = y
        # self.y = np.ones(shape=y.shape).astype(np.float32)
        sensitive_categories = sensitive.unique()
        # print(sencat)
        self.category_to_index_dict = dict(
            zip(list(sensitive_categories), range(len(sensitive_categories)))
        )
        self.index_to_category_dict = dict(
            zip(range(len(sensitive_categories)), list(sensitive_categories))
        )
        self.sensitive = sensitive
        self.sensitive_id = self.category_to_index_dict[self.sensitive]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].reshape(-1), self.sensitive_id[idx]


class TestConstraint:
    def test_costraint(self):
        consloss = ConstraintLoss()
        assert isinstance(consloss, ConstraintLoss)


class TestDemographicParityLoss:
    params = {"test_dp": [dict(feature_dim=16, sample_size=128, dim_condition=2)]}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def test_dp(self, feature_dim=16, sample_size=128, dim_condition=2):

        model = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        dp_loss = DemographicParityLoss(sensitive_classes=[0, 1])
        assert isinstance(dp_loss, DemographicParityLoss)

        x_train = torch.randn((sample_size, feature_dim))
        sensitive_features = torch.randint(0, dim_condition, (sample_size,))
        out = model(x_train)

        mu = dp_loss.mu_f(x_train, out, sensitive_features)
        assert int(mu.size(0)) == dim_condition + 1

        loss = dp_loss(x_train, out, sensitive_features)
        assert float(loss) >= 0

    def test_performance(self, feature_dim=16, sample_size=1280, dim_condition=2):
        x = torch.randn((sample_size, feature_dim))
        y = torch.randint(0, 2, (sample_size,))
        sensitive_features = torch.randint(0, dim_condition, (sample_size,))
        dataset = SensitiveDataset(x, y, sensitive_features)
        train_size = len(dataset)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, int(0.8 * train_size))

        model = nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 2))
        loss = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters())
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        model = self.__train_model(
            model=model, loss=loss, optimizer=optimizer, data_loader=train_loader
        )

    def __train_model(self, model, loss, data_loader, optimizer, max_epoch=100):
        for epoch in range(max_epoch):
            for i, data in enumerate(data_loader):
                optimizer.zero_grad()
                logit = model(data["X"].to(self.device))
                loss = loss(logit, data["y"].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type="inf")
                optimizer.step()
