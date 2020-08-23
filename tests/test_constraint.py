import unittest

import torch
from torch import nn

from fairtorch import ConstraintLoss, DemographicParityLoss


class TestConstraint(unittest.TestCase):
    def test_costraint(self):
        consloss = ConstraintLoss()
        self.assertTrue(isinstance(consloss, ConstraintLoss))

    def test_dp(self):
        fdim = 16
        model = nn.Sequential(nn.Linear(fdim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        dp_loss = DemographicParityLoss(sensitive_classes=[0,1])
        self.assertTrue(isinstance(dp_loss, DemographicParityLoss))
        bsize = 128
        n_A = 2
        X = torch.randn((bsize, fdim))
        y = torch.randint(0, 2, (bsize,))
        sensitive = torch.randint(0, n_A, (bsize,))
        out = model(X)

        mu = dp_loss.mu_f(X, out, sensitive)
        print(mu.size(), type(mu.size()))
        self.assertEqual(int(mu.size(0)), n_A + 1)

        loss = dp_loss(X, out, sensitive)

        self.assertGreater(float(loss), 0)


if __name__ == "__main__":
    unittest.main()
