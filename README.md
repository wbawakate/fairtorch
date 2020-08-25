# FairTorch

PyTorch implementation of parity loss as constraints function to realize the fairness of machine learning.

Demographic parity loss and equalied odds loss are available.

## Usage

```python
import torch
from torch.utils.data import DataLoader, Dataset
from fairtorch import DemographicParityLoss


class SensitiveDataset(Dataset):
    def __init__(self, x, y, sensitive):
        self.x = x
        self.y = y
        # self.y = np.ones(shape=y.shape).astype(np.float32)
        sensitive_categories = sensitive.unique().numpy()
        # print(sencat)
        self.category_to_index_dict = dict(
            zip(list(sensitive_categories), range(len(sensitive_categories)))
        )
        self.index_to_category_dict = dict(
            zip(range(len(sensitive_categories)), list(sensitive_categories))
        )
        self.sensitive = sensitive
        self.sensitive_ids = [
            self.category_to_index_dict[i] for i in self.sensitive.numpy().tolist()
        ]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.sensitive_ids[idx]


x = torch.randn((sample_size, feature_dim))
y = torch.randint(0, 2, (sample_size,))
sensitive_features = torch.randint(0, dim_condition, (sample_size,))
dataset = SensitiveDataset(x, y, sensitive_features)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

weight = 0.5
criterion = nn.CrossEntropyLoss()
constraints = weight * DemographicParityLoss()

model = nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = optim.Adam(model.parameters())

for epoch in range(max_epoch):
    for i, data in enumerate(data_loader):
        x, y, sensitive_features = data
        optimizer.zero_grad()
        logit = model(x.to(self.device))
        logit = model(x.to(self.device))
        loss = criterion(logit, y)
        penalty = constraints(x, logit, sensitive_features, y)
        loss = loss + penalty
        loss.backward()
        optimizer.step()
```

## Install

### pip version

```text
pip install fairtorch
```

### newest version

```text
git clone https://github.com/MasashiSode/fairtorch
cd fairtorch
pip install .
```

## Development

```text
poetry install
```

## Test

```text
pytest
```

## Authors

- [Akihiko Fukuchi](https://github.com/akiFQC)
- [Yoko Yabe](https://github.com/ykt345)
- [Masashi Sode](https://github.com/MasashiSode)
