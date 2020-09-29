#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
"""
@author: red0orange
@file: utils.py
@time:  上午10:49
@desc:
"""
import torch
import numpy as np
import os


def norm(v, lp):
    """单元测试成功"""
    if lp == 1:
        norms = (torch.sum(torch.abs(v), dim=[1, 2, 3]))[..., None, None, None]
    elif lp == 2:
        norms = (torch.sum(v ** 2, dim=[1, 2, 3])** (1 / 2.))[..., None, None, None]
    else:
        raise ValueError('wrong lp')
    return norms


def lp_project(x_adv, x_orig, eps, lp):
    """单元测试成功"""
    x_adv = torch.clamp(x_adv, 0., 1.)
    delta = x_adv - x_orig
    if lp == 2:
        norm_delta = norm(delta, lp=2)
        delta = delta / norm_delta * torch.min(eps, norm_delta, out=None)
    elif lp == np.inf:
        delta = torch.clamp(delta, -eps, eps)
    else:
        raise ValueError('wrong lp')
    return x_orig + delta


def gen_rubbish_permuted_images(x, y):
    """单元测试成功"""
    def permute_each(x_img):
        if len(list(x_img.size())) == 3:
            channels = x_img.size(0)
            x_flat = torch.reshape(x_img, [-1, channels])
        else:
            x_flat = torch.reshape(x_img, [-1])
        np.random.shuffle(x_flat)
        x_permuted = torch.reshape(x_flat, x_img.size())
        return x_permuted

    for i in range(x.size(0)):
        x[i] = permute_each(x_img=x[i])

    y = torch.ones_like(y) * 1.0 / float(y.size(1))
    return x, y


def gen_rubbish_uniform(x, y):
    """测试成功"""
    x = torch.Tensor(x.size()).uniform_()
    y = torch.ones_like(y) * 1.0 / float(y.size(1))
    return x, y


def rescale_to_zero_one(x):
    dims = [1, 2, 3]
    min_val = x
    max_val = x
    for dim in dims:
        min_val = torch.min(min_val, dim=dim, keepdim=True)[0]
        max_val = torch.max(max_val, dim=dim, keepdim=True)[0]
    x = (x - min_val) / (max_val - min_val)
    return x


def apply_proper_conv(x, full_kernel):
    n_pad = int(full_kernel.size(2))
    paddings = [[0, 0], [0, 0], [n_pad, n_pad], [n_pad, n_pad]]
    x_data = np.pad(x, paddings, mode='symmetric')
    x.data = torch.from_numpy(x_data)
    pad = ((full_kernel.size(2) - 1) // 2, (full_kernel.size(3) - 1) // 2)
    x = torch.nn.functional.conv2d(x, full_kernel, stride=[1, 1], padding=pad)
    x = x[:, :, n_pad:-n_pad, n_pad:-n_pad]
    x = rescale_to_zero_one(x)
    return x


def gaussian_kernel(std: float):
    """Makes 2D gaussian Kernel for convolution."""
    """单元测试成功"""
    size = 7
    mean = 0.0

    d = torch.distributions.Normal(mean, std)
    vals = torch.exp(d.log_prob(torch.arange(-size, size + 1)))
    gauss_kernel = torch.einsum('i,j->ij', vals, vals)
    return gauss_kernel / torch.sum(gauss_kernel)


def apply_random_lowpass(x):
    std = torch.Tensor(1).uniform_(1.0, 2.5)[0]
    gauss_kernel = gaussian_kernel(std)
    if x.size(1) == 1:
        full_kernel = gauss_kernel[None, None, :, :]
    else:  # if 3 colors
        zero_kernel = torch.zeros_like(gauss_kernel)
        kernel1 = torch.stack([gauss_kernel, zero_kernel, zero_kernel], dim=2)
        kernel2 = torch.stack([zero_kernel, gauss_kernel, zero_kernel], dim=2)
        kernel3 = torch.stack([zero_kernel, zero_kernel, gauss_kernel], dim=2)
        full_kernel = torch.stack([kernel1,kernel2,kernel3], dim=3)
    x = apply_proper_conv(x, full_kernel)
    return x


def elementwise_best(x, loss_elementwise, x_best, loss_best_elementwise):
    """单元测试成功"""
    take_prev = (loss_best_elementwise >= loss_elementwise).float()
    take_curr = (loss_best_elementwise < loss_elementwise).float()
    loss_best_elementwise = take_prev * loss_best_elementwise + take_curr * loss_elementwise
    x_best = torch.reshape(take_prev, [x.size(0), 1, 1, 1]) * x_best + torch.reshape(take_curr,[x.size(0), 1, 1, 1]) * x
    return x_best, loss_best_elementwise


def get_loss(logits, y, max_conf_flag_tf):
    maxclass = torch.argmax(logits, dim=-1)
    if max_conf_flag_tf:
        loss_elementwise = -torch.nn.functional.cross_entropy(logits, maxclass, reduce=False)
    else:
        loss_elementwise = torch.nn.functional.cross_entropy(logits, y, reduce=False)
    return loss_elementwise


def gen_adv_main(model, x, y, lp, eps, n_iters, step_size, max_conf_flag_tf):
    logits_x = model(x, flag_train=False)
    loss_x = get_loss(logits_x, torch.argmax(y,dim=-1), max_conf_flag_tf)

    # We experimented with other norms before, but eventually only the infinity pgd attack was used.
    assert lp == np.inf, 'Currently, only l-infinity attack is supported.'

    starting_perturbation = torch.Tensor(x.size(0), 1, 1, 1).uniform_(0.0, 1.0)
    unif = starting_perturbation * torch.Tensor(*x.size()).uniform_(-eps, eps)
    # unif = 0.0  # to remove the random step
    start_adv = torch.clamp(x + unif, 0., 1.)
    logits_start = model(start_adv, flag_train=False)
    loss_start = get_loss(logits_start, torch.argmax(y,dim=-1), max_conf_flag_tf)

    x_best_start, loss_best_start = elementwise_best(start_adv, loss_start, x, loss_x)

    initial_vars = [0, start_adv, x_best_start, loss_best_start]
    i, x_adv, x_best, loss_best = initial_vars
    while i < n_iters:
        # we never update BN averages during generation of adv. examples
        model.zero_grad()
        x_adv.requires_grad = True
        logits = model(x_adv, flag_train=False)
        loss = get_loss(logits, torch.argmax(y,dim=-1), max_conf_flag_tf)
        torch.sum(loss).backward()
        g = x_adv.grad
        g = torch.sign(g)
        x_adv = lp_project(x_adv + step_size * g, x, eps, lp).detach()
        logits_after_upd = model(x_adv, flag_train=False)
        loss_after_upd = get_loss(logits_after_upd, torch.argmax(y,dim=-1), max_conf_flag_tf)
        x_best, loss_best = elementwise_best(x_adv, loss_after_upd, x_best, loss_best)
        i += 1
    return x_best.detach(), y


def gen_adv(model, x, y, lp, eps, n_iters, step_size, rub_flag_tf, adv_flag_tf, frac_perm, apply_lowpass, max_conf_flag_tf):
    n_permuted = int(float(x.size(0)) * frac_perm)
    x_permuted, y_permuted = gen_rubbish_permuted_images(x[:n_permuted], y[:n_permuted])
    x_uniform, y_uniform = gen_rubbish_uniform(x[n_permuted:], y[n_permuted:])
    x_rubbish = torch.cat([x_permuted, x_uniform], dim=0)
    y_rubbish = torch.cat([y_permuted, y_uniform], dim=0)

    if apply_lowpass:
        x_rubbish = apply_random_lowpass(x_rubbish)

    x, y = (x_rubbish, y_rubbish) if rub_flag_tf else (x, y)
    if (x.size(0) > 0 and adv_flag_tf) and n_iters > 0:
        x_adv, y_adv = gen_adv_main(model, x, y, lp, eps, n_iters, step_size, max_conf_flag_tf)
    else:
        x_adv, y_adv = (x, y)
    model.zero_grad()   # 生成噪音图片结束后一切归零，只是生成了多一部分的batch_x
    return x_adv, y_adv


def get_trainable_grad_and_var(named_params):
    result = []
    for name, param in named_params:
        if param.requires_grad:
            result.append([name, param.grad, param.data])
    return result


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    """
    # 只管单gpu的情况，不管多gpu的，均值直接就是原来的值
    average_grads = {}
    for per_gpu_grad in tower_grads:
        for (name, grad, var) in per_gpu_grad:
            average_grads[name] = (grad, var)
    return average_grads


def apply_gradients(model, grads:dict):
    """
    将梯度手动赋值给模型
    @param model:
    @type model:
    @param grads: (name, grad, var)
    @type grads:
    @return:
    @rtype:
    """
    for name, param in model.named_parameters():
        if grads.get(name) is None:
            raise BaseException('不对应的梯度')
        param.grad = grads[name][0]
