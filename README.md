# FairTorch

Implementation of parity loss as constraints function to realize the fairness of machine learning.

Demographic parity loss and equalied odds loss are available.

## Usage

```python
from fairtorch import DemographicParityLoss
weight = 0.5
criterion = nn.CrossEntropyLoss() + weight * DemographicParityLoss()
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
