
def count_params(module):
    """Print total number of parameteres in given module"""
    num_params = 0
    for param in module.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def view_params(module):
    """Overview all parameters of given model (model name, param size) + # of total params""""
    print('Overview Parameters')
    num_params = 0
    for name, param in self.model.named_parameters():
        print('\t' + name + '\t', list(param.size()))
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
