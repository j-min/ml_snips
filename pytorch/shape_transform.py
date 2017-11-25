import torch
from torch.autograd import Variable

def reverse_order(tensor, dim=0):
    """Reverse Tensor or Variable along given dimension"""
    if isinstance(tensor, torch.Tensor) or isinstance(tensor, torch.LongTensor):
        idx = [i for i in range(tensor.size(dim)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = tensor.index_select(dim, idx)
        return inverted_tensor
    elif isinstance(tensor, Variable):
        tensor.data = reverse_order(tensor.data, dim=dim)
        return tensor
