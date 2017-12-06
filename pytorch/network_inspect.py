def count_parameters(module):
    """Print total number of parameteres in given module"""
    num_params = 0
    for param in module.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
