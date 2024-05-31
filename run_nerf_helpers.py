import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import matplotlib.pyplot as plt

# Misc utils
def show_colorbar(image, cmap='rainbow'):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.cpu().numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf

def img2mse(x, y):
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log10(x)

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def define_image_grid_3D_np(x_size, y_size):
    y = np.arange(x_size)
    x = np.arange(y_size)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    image_grid = np.vstack((image_grid_xy, z))
    return image_grid

# Positional encoding
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        B = self.kwargs['B']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if B is not None:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(x @ B.T * freq))
                    out_dim += d + B.shape[1]
                else:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                    out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, b=0):
    if i == -1:
        return nn.Identity(), 3
    if b != 0:
        B = torch.randn((b, 3))
    else:
        B = None

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'B': B
    }
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

# Model architecture
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=6, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for l in self.views_linears:
                h = F.relu(l(h))

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs

# Ray helpers
def get_rays_us_linear(H, W, sw, sh, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    x = torch.arange(-W / 2, W / 2, dtype=torch.float32) * sw
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)

    origin_base = torch.stack([x, y, z], dim=1)
    origin_base_prim = origin_base[..., None, :]
    origin_rotated = R * origin_base_prim
    ray_o_r = torch.sum(origin_rotated, dim=-1)
    rays_o = ray_o_r + t

    dirs_base = torch.tensor([0., 1., 0.])
    dirs_r = torch.matmul(R, dirs_base)
    rays_d = dirs_r.expand_as(rays_o)

    return rays_o, rays_d

# Hierarchical sampling helper
def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0., 1., N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)
    cdf_g = torch.gather(cdf, dim=-1, index=inds_g)
    bins_g = torch.gather(bins, dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
