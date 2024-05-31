import numpy as np
import torch
import torch.nn.functional as F
import math

EPS = 1.0e-7

def mutualInformation(bin_centers,
                      sigma_ratio=0.5,
                      max_clip=1,
                      crop_background=False,
                      local_mi=False,
                      patch_size=1,
                      vol_size=None):
    if local_mi:
        return localMutualInformation(bin_centers, vol_size, sigma_ratio, max_clip, patch_size)
    else:
        return globalMutualInformation(bin_centers, sigma_ratio, max_clip, crop_background)

def globalMutualInformation(bin_centers,
                            sigma_ratio=0.5,
                            max_clip=1,
                            crop_background=False):
    vol_bin_centers = torch.tensor(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
    preterm = torch.tensor(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0, max_clip)
        y_true = torch.clamp(y_true, 0, max_clip)

        if crop_background:
            thresh = 0.0001
            padding_size = 20
            filt = torch.ones(1, 1, padding_size, padding_size, padding_size).to(y_true.device)
            smooth = F.conv3d(y_true, filt, padding=padding_size//2)
            mask = smooth > thresh
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            y_pred = y_pred.view(1, -1, 1)
            y_true = y_true.view(1, -1, 1)
        else:
            y_true = y_true.view(y_true.shape[0], -1, 1)
            y_pred = y_pred.view(y_pred.shape[0], -1, 1)

        nb_voxels = torch.tensor(y_pred.shape[1], dtype=torch.float32, device=y_pred.device)
        vbc = vol_bin_centers.view(1, 1, num_bins)

        I_a = torch.exp(- preterm * torch.square(y_true - vbc))
        I_a /= torch.sum(I_a, -1, keepdim=True)

        I_b = torch.exp(- preterm * torch.square(y_pred - vbc))
        I_b /= torch.sum(I_b, -1, keepdim=True)

        I_a_permute = I_a.permute(0, 2, 1)
        pab = torch.bmm(I_a_permute, I_b) / nb_voxels
        pa = torch.mean(I_a, 1, keepdim=True)
        pb = torch.mean(I_b, 1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + EPS
        mi = torch.sum(pab * torch.log(pab / papb + EPS), dim=[1, 2])

        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred).mean()

    return loss

def localMutualInformation(bin_centers, vol_size, sigma_ratio=0.5, max_clip=1, patch_size=1):
    vol_bin_centers = torch.tensor(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
    preterm = torch.tensor(1 / (2 * np.square(sigma)))

    def local_mi(y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0, max_clip)
        y_true = torch.clamp(y_true, 0, max_clip)
        vbc = vol_bin_centers.view(1, 1, 1, 1, num_bins)

        x, y, z = vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [(x_r // 2, x_r - x_r // 2), (y_r // 2, y_r - y_r // 2), (z_r // 2, z_r - z_r // 2)]
        padding = [(0, 0), *pad_dims, (0, 0)]
        padding = [item for sublist in padding for item in sublist]

        I_a = torch.exp(- preterm * torch.square(F.pad(y_true, padding, 'constant') - vbc))
        I_a /= torch.sum(I_a, -1, keepdim=True)

        I_b = torch.exp(- preterm * torch.square(F.pad(y_pred, padding, 'constant') - vbc))
        I_b /= torch.sum(I_b, -1, keepdim=True)

        I_a_patch = I_a.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
        I_a_patch = I_a_patch.permute(0, 2, 3, 4, 1, 5, 6).reshape(-1, patch_size ** 3, num_bins)

        I_b_patch = I_b.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
        I_b_patch = I_b_patch.permute(0, 2, 3, 4, 1, 5, 6).reshape(-1, patch_size ** 3, num_bins)

        I_a_permute = I_a_patch.permute(0, 2, 1)
        pab = torch.bmm(I_a_permute, I_b_patch) / (patch_size ** 3)
        pa = torch.mean(I_a_patch, 1, keepdim=True)
        pb = torch.mean(I_b_patch, 1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + EPS
        mi = torch.mean(torch.sum(pab * torch.log(pab / papb + EPS), dim=[1, 2]))

        return mi

    def loss(y_true, y_pred):
        return -local_mi(y_true, y_pred)

    return loss

def fit_kmeans_sklearn(image_data, n_clusters=5):
    from sklearn.cluster import KMeans
    image_data = image_data.reshape((-1, 1))
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=1, max_iter=5)
    k_means.fit(image_data)
    return k_means

def fit_kmeans_tensor(image_data, k_means, num_iterations=100):
    image_data = image_data.numpy().reshape((-1, 1))
    for _ in range(num_iterations):
        k_means.partial_fit(image_data)
    labels = k_means.predict(image_data)
    cluster_centers = k_means.cluster_centers_
    return labels.reshape(image_data.shape), cluster_centers

def fit_gmm(image_data, n_components=6):
    from sklearn.mixture import GaussianMixture as GMM
    org_shape = image_data.shape
    image_data = image_data.reshape((-1, 1))
    gmm_model = GMM(n_components=n_components, covariance_type='full').fit(image_data)
    gmm_labels = gmm_model.predict(image_data)
    gmm_labels = gmm_labels.reshape(org_shape)
    return gmm_labels, gmm_model.means_

def separable_filter(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Create a 3d separable filter in PyTorch.
    :param tensor: shape = (batch, dim1, dim2, dim3, 1)
    :param kernel: shape = (dim4,)
    :return: shape = (batch, dim1, dim2, dim3, 1)
    """
    kernel = kernel.to(tensor.dtype)
    kernel = kernel.view(-1, 1, 1, 1, 1)

    tensor = F.conv3d(tensor, kernel, padding=0)
    kernel = kernel.view(1, -1, 1, 1, 1)
    tensor = F.conv3d(tensor, kernel, padding=0)
    kernel = kernel.view(1, 1, -1, 1, 1)
    tensor = F.conv3d(tensor, kernel, padding=0)

    return tensor

def rectangular_kernel1d(kernel_size: int) -> torch.Tensor:
    """
    Return a 1D rectangular kernel for LocalNormalizedCrossCorrelation.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    kernel = torch.ones(kernel_size, dtype=torch.float32)
    return kernel

def triangular_kernel1d(kernel_size: int) -> torch.Tensor:
    """
    Return a 1D triangular kernel for LocalNormalizedCrossCorrelation.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    padding = kernel_size // 2
    kernel = torch.tensor(
        [0] * math.ceil(padding / 2) + [1] * (kernel_size - padding) + [0] * math.floor(padding / 2),
        dtype=torch.float32
    )

    filters = torch.ones((kernel_size - padding, 1, 1), dtype=torch.float32)

    kernel = F.conv1d(kernel.view(1, 1, -1), filters, padding="same")
    return kernel.view(-1)

def gaussian_kernel1d(kernel_size: int) -> torch.Tensor:
    """
    Return a 1D Gaussian kernel for LocalNormalizedCrossCorrelation.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: filters, of shape (kernel_size, )
    """
    mean = (kernel_size - 1) / 2.0
    sigma = kernel_size / 3

    grid = torch.arange(0, kernel_size, dtype=torch.float32)
    filters = torch.exp(-torch.square(grid - mean) / (2 * sigma ** 2))

    return filters

def gaussian_kernel1d_sigma(sigma: int) -> torch.Tensor:
    """
    Calculate a Gaussian kernel.
    :param sigma: number defining standard deviation for Gaussian kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    tail = int(sigma * 3)
    kernel = torch.exp(-0.5 * torch.square(torch.arange(-tail, tail + 1, dtype=torch.float32)) / sigma ** 2)
    kernel = kernel / torch.sum(kernel)
    return kernel

def cauchy_kernel1d(sigma: int) -> torch.Tensor:
    """
    Approximating Cauchy kernel in 1D.
    :param sigma: int, defining standard deviation of kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    tail = int(sigma * 5)
    k = 1 / ((torch.arange(-tail, tail + 1, dtype=torch.float32) / sigma) ** 2 + 1)
    k = k / torch.sum(k)
    return k

def _hgram(img_l, img_r):
    hgram, _, _ = np.histogram2d(img_l.ravel(), img_r.ravel())
    return hgram

def _mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mi(img_l, img_r):
    return _mutual_information(_hgram(img_l, img_r))
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, signed=False):
        self.win = win
        self.eps = eps
        self.signed = signed

    def ncc(self, Ii, Ji):
        # get dimension of volume
        ndims = Ii.ndim - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.shape[-1]
        sum_filt = torch.ones([1, 1, *self.win]).to(Ii.device) / np.prod(self.win)
        strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'same'
        I_sum = F.conv2d(Ii.permute(0, 4, 1, 2, 3).reshape(-1, 1, *Ii.shape[1:-1]), sum_filt, padding=padding).reshape(Ii.shape[0], -1, *Ii.shape[1:-1]).permute(0, 2, 3, 4, 1)
        J_sum = F.conv2d(Ji.permute(0, 4, 1, 2, 3).reshape(-1, 1, *Ji.shape[1:-1]), sum_filt, padding=padding).reshape(Ji.shape[0], -1, *Ji.shape[1:-1]).permute(0, 2, 3, 4, 1)
        I2_sum = F.conv2d(I2.permute(0, 4, 1, 2, 3).reshape(-1, 1, *I2.shape[1:-1]), sum_filt, padding=padding).reshape(I2.shape[0], -1, *I2.shape[1:-1]).permute(0, 2, 3, 4, 1)
        J2_sum = F.conv2d(J2.permute(0, 4, 1, 2, 3).reshape(-1, 1, *J2.shape[1:-1]), sum_filt, padding=padding).reshape(J2.shape[0], -1, *J2.shape[1:-1]).permute(0, 2, 3, 4, 1)
        IJ_sum = F.conv2d(IJ.permute(0, 4, 1, 2, 3).reshape(-1, 1, *IJ.shape[1:-1]), sum_filt, padding=padding).reshape(IJ.shape[0], -1, *IJ.shape[1:-1]).permute(0, 2, 3, 4, 1)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, torch.tensor(self.eps, device=cross.device))
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, torch.tensor(self.eps, device=I_var.device))
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, torch.tensor(self.eps, device=J_var.device))

        if self.signed:
            cc = cross / torch.sqrt(I_var * J_var + self.eps)
        else:
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            cc = torch.mean(cc)
        elif reduce == 'max':
            cc = torch.max(cc)
        elif reduce is not None:
            raise ValueError(f'Unknown NCC reduction type: {reduce}')
        # loss
        return -cc

class LocalNormalizedCrossCorrelation(torch.nn.Module):
    """
    Local squared zero-normalized cross-correlation.
    """

    kernel_fn_dict = {
        'gaussian': gaussian_kernel1d,
        'rectangular': rectangular_kernel1d,
        'triangular': triangular_kernel1d,
    }

    def __init__(self, kernel_size: int = 9, kernel_type: str = "rectangular",
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5):
        super().__init__()
        if kernel_type not in self.kernel_fn_dict:
            raise ValueError(
                f"Wrong kernel_type {kernel_type} for LNCC loss type. "
                f"Feasible values are {list(self.kernel_fn_dict.keys())}"
            )
        self.kernel_fn = self.kernel_fn_dict[kernel_type]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

        self.kernel = self.kernel_fn(kernel_size=self.kernel_size)
        self.kernel_vol = torch.sum(
            self.kernel[:, None, None]
            * self.kernel[None, :, None]
            * self.kernel[None, None, :]
        )

    def calc_ncc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        t2 = y_true * y_true
        p2 = y_pred * y_pred
        tp = y_true * y_pred

        t_sum = separable_filter(y_true, kernel=self.kernel)
        p_sum = separable_filter(y_pred, kernel=self.kernel)
        t2_sum = separable_filter(t2, kernel=self.kernel)
        p2_sum = separable_filter(p2, kernel=self.kernel)
        tp_sum = separable_filter(tp, kernel=self.kernel)

        t_avg = t_sum / self.kernel_vol
        p_avg = p_sum / self.kernel_vol

        cross = tp_sum - p_avg * t_sum
        t_var = t2_sum - t_avg * t_sum
        p_var = p2_sum - p_avg * p_sum

        t_var = torch.clamp(t_var, min=0)
        p_var = torch.clamp(p_var, min=0)

        ncc = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)

        return ncc

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if y_true.ndim == 4:
            y_true = y_true.unsqueeze(4)
        if y_true.shape[4] != 1:
            raise ValueError(f"Last dimension of y_true is not one. y_true.shape = {y_true.shape}")

        if y_pred.ndim == 4:
            y_pred = y_pred.unsqueeze(4)
        if y_pred.shape[4] != 1:
            raise ValueError(f"Last dimension of y_pred is not one. y_pred.shape = {y_pred.shape}")

        ncc = self.calc_ncc(y_true, y_pred)
        return torch.mean(ncc, dim=[1, 2, 3, 4])

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "kernel_type": self.kernel_type,
            "smooth_nr": self.smooth_nr,
            "smooth_dr": self.smooth_dr,
        }
        return config
