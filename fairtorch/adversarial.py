import torch
from torch import nn
from torch.nn import functional as F
from torch import optim



class AdversaryNet(nn.Module):

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
        dim_output: int, default 1
            dimention of outpuy layers
        """
        super(AdversaryNet, self).__init__()
        self.n_layers = kargs["n_layers"] if "n_layers" in kargs else 1
        self.dim_input = kargs["dim_input"] if "dim_input" in kargs else 1
        self.dim_hidden = kargs["dim_hidden"] if "dim_hidden" in kargs else 16
        self.dim_output = kargs["dim_output"] if "dim_output" in kargs else 1

        self.nets = nn.ModuleList([])
        if self.n_layers == 1:
            self.nets.append(nn.Linear(self.dim_input, self.dim_output) )
        else:
            for i in range(1, self.n_layers + 1):
                if i == self.n_layers:
                    dim_out = self.dim_output
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
            x = self.nets[i].forward(x)
        return x

        
class AdversarialDebiasingLoss(nn.Module):

    def __init__(self, **kargs):
        """
        Adversarial Debiasing Loss

        Parameters:
        --------------
        parity: str, default "demographic_parity"
            parity criterion. choose from ["deomographic_parity", "DP",  "equalized_odds", "EO"]
        n_layers: int, default 1
            number of layers
        dim_hidden: int, default 16
            dimention of hidden layers
        sensitive_classes: list,  defualt [0, 1]
            class labels of sensitive label
        n_iter : int, default 1
            number of iteration for training adversary net
        alpha : float, default 1.0
            coefficient of adverarial loss
        """
        super(AdversarialDebiasingLoss, self).__init__()
        self.n_layers = kargs["n_layers"] if "n_layers" in kargs else 1
        self.dim_hidden = kargs["dim_hidden"] if "dim_hidden" in kargs else 16
        self.sensitive_classes = kargs["sensitive_classes"] if "sensitive_classes" in kargs else [0, 1]
        self.class_map = {}
        for i, sc in enumerate(self.sensitive_classes):
            self.class_map[sc] = i
        self.n_classes = len(self.sensitive_classes)
        # print("class map", self.class_map)
        if not  "parity" in kargs:
            self.parity = "DP"
            self.dim_input = 1
        elif kargs["parity"] in ("equalized_odds",  "EO"):
            self.parity = "EO"
            self.dim_input = 2
        elif kargs["parity"] in ("demographic_parity",  "DP"):
            self.parity = "DP"
            self.dim_input = 1
        else : 
            raise ValueError(f'"parity" must be selected from ["deomographic_parity", "DP",  "equalized_odds", "EO"], but got {kargs["parity"]}')
        # adversary network
        self.adversary_net = AdversaryNet(
            n_layers=self.n_layers, 
            dim_input=self.dim_input, 
            dim_hidden=self.dim_hidden,
            dim_output=self.n_classes,
        )
        if "device" in kargs:
            self.device = kargs["device"] 
        elif torch.cuda.is_available():
            self.device = "cuda"
        else :
            self.device = "cpu"
        self.alpha = kargs['alpha'] if 'alpha' in kargs else 1.0
        self.adversary_net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.adversary_net.parameters())
        self.n_iter = kargs["n_iter"] if "n_iter" in kargs else 1

    def forward(self, X, out, sensitive, y=None):
        # print(sensitive)
        if isinstance(sensitive, torch.Tensor):
            sensitive_ids = sensitive
            sensitive_tensor = sensitive_ids
        else:
            sensitive_ids = [self.class_map[s] for s in sensitive]
            sensitive_tensor = torch.Tensor(sensitive_ids).long().to(X.device)
        out = torch.sigmoid(out)
        if self.parity == "DP":
            input2adv = out
        else:
            # print("shapes", out.shape, y.shape)
            if out.shape != y.shape:
                y = y.reshape(out.shape)
            input2adv = torch.cat([out, y ], dim=1)
        # update adversary net
        self.optimizer.zero_grad()
        for i in range(self.n_iter):
            out0 = self.adversary_net.forward(input2adv.clone().detach())
            loss0 = self.criterion(out0, sensitive_tensor.clone().detach())
            self.optimizer.zero_grad()
            loss0.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        # make adversary
        out_adv = self.adversary_net.forward(input2adv)
        loss_adv = self.criterion(out_adv, sensitive_tensor)
        return -1* self.alpha * loss_adv

