import torch
from torch.autograd import Variable

def reverse_order(tensor):
    if isinstance(tensor, torch.Tensor):
        idx = [i for i in range(tensor.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor
    elif isinstance(tensor, Variable):
        tensor.data = reverse_order(tensor.data)
        return tensor
