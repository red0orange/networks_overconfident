import torch


def weight_decay(named_params, norm=2):
    costs = []
    trainable_variables = []
    for name, param in named_params:
        if param.requires_grad:
            trainable_variables.append([name, param])
    for name, var in trainable_variables:
        if 'conv' in name or 'fc' in name or 'weights' in name:
            if norm == 1:
                lp_norm_var = torch.sum(torch.abs(var))
            elif norm == 2:
                lp_norm_var = torch.sum(var ** 2)
            else:
                raise ValueError('wrong norm of weight decay')
            costs.append(lp_norm_var)
    return sum(costs)

