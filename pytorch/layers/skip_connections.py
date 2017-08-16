from torch import nn
from torch.nn import functional as F

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
        Args:
            x  (Tensor): [batch_size, size]
        Return:
            x' (Tensor): [batch_size, size]
            
        Applies σ(x) ⨀ f(G(x)) + (1 - σ(x)) ⨀ Q(x)
        
        linear: Q (affine transformation)
        nonlinear: f (non-linear tranformation) with G (affine transformation)
        gate: σ(x) (affine transformation) with sigmoid
        ⨀: element-wise multiplication
        """

        for nonlinear, linear, gate in zip(self.nonlinear, self.linear, self.gate):
            gate = F.sigmoid(gate(x))
            x = gate * self.f(nonlinear(x)) + (1 - gate) * linear(x)

        return x
