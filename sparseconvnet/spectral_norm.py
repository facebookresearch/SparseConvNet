import torch

def spectral_norm(module, n_power_iterations=1, eps=1e-12):
    """
    https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/spectral_norm.py
    """
    dim=1
    torch.nn.utils.SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module
