import torch.nn as nn
import torch.nn.functional as F


class SamePad2d(nn.Module):
    def __init__(self, kernel_size, stride, value=0):
        """"SAME-Padding Wrapper for PyTorch"""
        super(SamePad2d, self).__init__()

        if type(kernel_size) == int:
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size
        elif len(kernel_size) == 2:
            self.kernel_width = kernel_size[0]
            self.kernel_height = kernel_size[1]

        if type(stride) == int:
            self.stride_width = stride
            self.stride_height = stride
        elif len(stride) == 2:
            self.stride_width = stride[0]
            self.stride_height = stride[1]

        self.value = value

    def forward(self, x):
        batch_size, channel_size, height, width = x.size()

        width_residual = width % self.stride_width
        height_residual = height % self.stride_height

        if width_residual == 0:
            pad_along_width = max(self.kernel_width - self.stride_width, 0)
        else:
            pad_along_width = max(self.kernel_width - width_residual, 0)

        if height_residual == 0:
            pad_along_height = max(self.kernel_height - self.stride_height, 0)
        else:
            pad_along_height = max(self.kernel_height - height_residual, 0)

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top

        return F.pad(x, pad=(pad_left, pad_right, pad_top, pad_bottom), value=self.value)


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    x = Variable(torch.ones(1, 1, 5, 5))
    conv = nn.Conv2d(1, 1, 3)
    pad = SamePad2d(3, 1)
    print('x')
    print(x)
    print('pad(x)')
    print(pad(x))
    print('conv(pad(x))')
    print(conv(pad(x)))
