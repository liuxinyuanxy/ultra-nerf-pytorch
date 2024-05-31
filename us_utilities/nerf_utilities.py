import torch
import torch.nn.functional as F
import numpy as np


def patch_l2(y, y_prim):
    patches_y = F.unfold(y.unsqueeze(0).unsqueeze(0), kernel_size=(8, 8), stride=(1, 1))
    patches_y_prim = F.unfold(y_prim.unsqueeze(0).unsqueeze(0), kernel_size=(8, 8), stride=(1, 1))
    patch_sum_y = patches_y.sum(dim=1, keepdim=True)
    patch_sum_y_prim = patches_y_prim.sum(dim=1, keepdim=True)

    return img2mse(patch_sum_y.squeeze(), patch_sum_y_prim.squeeze())


def compute_tv_norm(values, losstype='l2', weighting=None):
    """Returns TV norm for input values."""
    v00 = values[:-1, :-1]
    v01 = values[:-1, 1:]
    v10 = values[1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')

    if weighting is not None:
        loss = loss * weighting

    return loss


def gaussian_kernel(size: int, mean: float, std: float):
    delta_t = 1  # 9.197324e-01
    x_cos = np.arange(-size, size + 1, dtype=np.float32) * delta_t

    y_modulation = torch.cos(torch.tensor(x_cos) * 2 * np.pi * 8e6)
    d1 = torch.distributions.Normal(mean, std * 3.)
    d2 = torch.distributions.Normal(mean, std)

    vals_x = d1.log_prob(torch.arange(-size, size + 1, dtype=torch.float32) * delta_t).exp()
    vals_y = d2.log_prob(torch.arange(-size, size + 1, dtype=torch.float32) * delta_t).exp()

    gauss_kernel = torch.ger(vals_x, vals_y)

    return gauss_kernel / gauss_kernel.sum()


# Helper function for mean squared error
def img2mse(x, y):
    return torch.mean((x - y) ** 2)
