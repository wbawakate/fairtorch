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
constraints = DemographicParityLoss()

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
        loss = loss + weight * penalty
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

## Background

In recent years, machine learning-based algorithms and softwares have rapidly spread in society. However, cases have been reported where these algorithms unintentionally suggest discriminatory decisions$^{1}$. For example, allocation harms can occur when AI systems extend or withhold opportunities, resources, or information. Some of the key applications are in hiring, school admissions, and lending$^2$. Since Pytorch didn't have a library to achieve fairness yet, we decided to create one.

## What it does

Fairtorch provides tools to mitigate inequities in classification and regression. Classification is only available in binary classification. A unique feature of this tool is that you can add a fairness constraint to your model by simply adding a few lines

## Challenges we ran into

In the beginning, we attempted to develop FairTorch based on the [Fairlearn](https://github.com/fairlearn/fairlearn)$^3$’s reduction algorithm. However, it was implemented based on scikit-learn and was not a suitable algorithm for deep learning. It requires ensemble training of the model, which would be too computationally expensive to be used for deep learning. To solve that problem, we implemented a constrained optimization without ensemble learning to fit the existing Fairlearn algorithm for deep learning.

## How we built it

We employ a method called group fairness, which is formulated by a constraint on the predictor's behavior called a parity constraint, where $X$ is the feature vector used for prediction, $A$ is a single sensitive feature (such as age or race), and $Y$ is the true label. A parity constraint is expressed in terms of an expected value about the distribution on $(X, A, Y)$.

In order to achieve the above, constrained optimization is adopted. We implemented loss as a constraint. The loss corresponds to parity constraints.

Demographic Parity and Equalized Odds are applied to the classification algorithm.

We consider a binary classification setting where the training
examples consist of triples $(X, A, Y)$, where X is a feature value, $A$ is a protected attribute, and $Y \in {0, 1}$ is a label.A classifier that predicts $Y$ from $X$ is $h: X→Y$.

The demographic parity is shown below.

$$
E[h(X)| A=a] = E[h(X)] \ \rm{for \ all} \ a∈A ・・・(1)
$$

Next, the equalized odds are shown below.

$$
E[h(X)| A=a、Y=y] = E[h(X)|Y=y] \ \rm{for \ all} \  a∈A, \  y∈Y・・・(2)
$$

We consider learning a classifier $h(X; \theta)$ by pytorch that satisfies these fairness conditions.
The \theta is a parameter. As an inequality-constrained optimization problem, we convert (1) and (2) to inequalities in order to train the classifier.

$$
M \mu (X, Y, A, h(X, \theta)) \leq c・・・(3)
$$

Thus, the study of the classifier $h(X; \theta)$ is as follows.
$\rm{Min}_{\theta}$s error$(X, Y)$ subject to $M$ $\mu (X, Y, A, h(X, \theta)) \leq c$
To apply this problem to pytorch's gradient method-based parameter optimization, we make the inequality constraint a constraint term R.

$$
R = B \cdot |ReLU(M \mu (X, Y, A, h(X, \theta)) - c)|^2・・・ (4)
$$

## Accomplishments that we're proud of

We confirmed by experiment that inequality is reduced just adding 2 lines of code.

## What We learned

We learned the definitions of fairness and the mathematical formulations to achieve it.

## What's next for FairTorch

As the current optimization algorithm is not yet refined in FairTorch, we plan to implement a more efficient constrained optimization algorithm. Other definitions of fairness have also been proposed besides demographic parity and equalized odds. In the future, we intend to implement other kinds of fairness.

## References

1. the keynote by K. Crawford at NeurIPS 2017
2. A Reductions Approach to Fair Classification (Alekh Agarwal et al.,2018)
3. Fairlearn: A toolkit for assessing and improving fairness in AI (Bird et al., 2020)

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
