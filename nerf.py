"""
nerf
====
Provides
* PE
    --- used in this project as a part of the network structure,
        and the type is `torch.nn.Module`
    --- **using high frequency functions before passing them to the
    network enables better fitting of data that contains high frequency variation.**
* NeRF
    --- Neural network implemented by the PyTorch, 
    whose network structure is roughly like Multiple Layer Perceptron (MLP), 
    consisting of several fully connected layers
* IPE
    --- integrated positional encoding in paper
* Mip-NeRF
    --- mip-NeRF implemented by the PyTorch,
    using IPE and ConeTracing
-------------------------------------------------------------------------------------
Author: LT H
Github: mofashaoye
"""
import numpy as np
import torch
from torch import nn


class _Encoder(nn.Module):
    """
    _Encoder
    ========
    Implementation of Positional encoding in the paper

    $F_\Theta(p)=(p,\sin(2^0\pi{p}),\cos(2^0\pi{p}),\cdots,\sin(2^{L-1}\pi{p}),\cos(2^{L-1}\pi{p}))$
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.func_list = [lambda x: x]
        for L in range(dim):
            self.func_list.append(lambda x, L=L: torch.sin(2.0**L * x))
            self.func_list.append(lambda x, L=L: torch.cos(2.0**L * x))

    def forward(self, x) -> torch.Tensor:
        return torch.cat([func(x) for func in self.func_list], -1)

    @property
    def out_features(self):
        return len(self.func_list)


class _NeRF(nn.Module):
    """
    _NeRF
    =====
    Neural network implemented by the PyTorch, 
    whose network structure is roughly like Multiple Layer Perceptron (MLP), 
    consisting of several fully connected layers
    MLP([...,inpos_features+inview_features]) = [..., 4]
    """

    def __init__(self, inpos_features=3, inview_features=3, dense_features=256, dense_depth=8, skips=[4]) -> None:
        super().__init__()

        self.skips = skips
        self.inpos_features = inpos_features
        self.inview_features = inview_features

        def dense(in_features, out_features, act=nn.ReLU()):
            return nn.Sequential(
                nn.Linear(in_features, out_features), act
            )
        # Some fully connected layers
        self.density_dense = nn.ModuleList(
            [dense(inpos_features, dense_features)] +
            [dense(dense_features, dense_features)
             if skips is None or i not in skips
             else dense(dense_features + inpos_features, dense_features)
             for i in range(dense_depth - 1)]
        )
        # Used to output temporary features
        self.feature_output = nn.Linear(dense_features, dense_features)
        # Used to output density
        self.density_output = nn.Linear(dense_features, 1)
        # Used to output radiance
        self.radiance_output = nn.Sequential(
            nn.Linear(dense_features + inview_features, dense_features // 2),
            nn.ReLU(),
            nn.Linear(dense_features // 2, 3)
        )

    def forward(self, x) -> torch.Tensor:
        pos_locs, view_dirs = torch.split(
            x, (self.inpos_features, self.inview_features), dim=-1
        )

        x = pos_locs
        for i, layer in enumerate(self.density_dense):
            x = layer(x)
            if i in self.skips:
                x = torch.cat((pos_locs, x), dim=-1)
        density = self.density_output(x)
        radiance = self.radiance_output(
            torch.cat((self.feature_output(x), view_dirs), -1)
        )
        return torch.cat((radiance, density), -1)


class MipNeRF(nn.Module):
    r"""
    NeRF
    ====
    * Complete Geometric Model for rendering
    * Complete Neural Network for training
    """
    COARSE = "coarse"   # Coarse Net Flag
    FINE = "fine"       # Fine   Net Flag

    def __init__(self, pos_dim=10, view_dim=8, dense_features=256, dense_depth=8, dense_features_fine=256, dense_depth_fine=8) -> None:
        super().__init__()

        # positional encoding in paper
        self.pos_encoder = _Encoder(dim=pos_dim)
        self.view_encoder = _Encoder(dim=view_dim)
        # coarse net
        self.coarse_model = _NeRF(
            inpos_features=self.pos_encoder.out_features * 3,
            inview_features=self.view_encoder.out_features * 3,
            dense_features=dense_features,
            dense_depth=dense_depth,
            skips=[4]
        )
        # fine net
        self.fine_model = _NeRF(
            inpos_features=self.pos_encoder.out_features * 3,
            inview_features=self.view_encoder.out_features * 3,
            dense_features=dense_features_fine,
            dense_depth=dense_depth_fine,
        )

        self.coarse_()

    def coarse_(self):
        return self.net_(self.COARSE)

    def fine_(self):
        return self.net_(self.FINE)

    def net_(self, type):
        """
        select which net to forward
        """
        self.forward_model = type
        return self

    def forward(self, x) -> torch.Tensor:
        """
        $p(x,y,z,d_x,d_y,d_z)\stackrel{Encoder}{\Longrightarrow}(p,\sin(2^0\pi{p}),\cos(2^0\pi{p}),\cdots,\sin(2^{L-1}\pi{p}),\cos(2^{L-1}\pi{p}))\stackrel{Coarse\,or\,{Fine}}{\Longrightarrow}{(r,g,b,\sigma)}$
        """
        pos_locs, view_dirs = torch.split(
            x, (3, 3), dim=-1
        )
        pos_locs = self.pos_encoder(pos_locs)
        view_dirs = self.view_encoder(view_dirs)
        x = torch.cat((pos_locs, view_dirs), -1)
        model = self.coarse_model if self.forward_model == self.COARSE else self.fine_model
        return model(x)


def safe_trig_helper(x, fn, t=100 * torch.tensor(np.pi)):
    t = t.to(x.device)
    return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_cos(x):
    return safe_trig_helper(x, torch.cos)


def safe_sin(x):
    return safe_trig_helper(x, torch.sin)


def expected_sin(x: torch.Tensor, x_var: torch.Tensor):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * safe_sin(x)
    y_var = torch.max(
        torch.tensor(0.0, device=x.device), 
        0.5 * (1 - torch.exp(-2 * x_var) * safe_cos(2 * x)) - y**2
    )
    return y, y_var


class _IEncoder(nn.Module):
    """
    _IEncoder
    =========
    Implementation of Integrated Positional Encoding
    """
    
    def __init__(self, max_dim, min_dim=0) -> None:
        super().__init__()
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.scales = torch.tensor(
            [2**i for i in range(self.min_dim, self.max_dim)]   
        )
        self.scales = nn.parameter.Parameter(self.scales, requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, x_cov_diag = inputs[..., :3], inputs[..., 3:] #[..., 3]
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * self.scales[:, None], shape) 
        # [..., 1, 3] * [max_dim - min_dim, 1] = [..., max_dim - min_dim, 3] 
        # => [..., (max_dim - min_dim) * 3]
        y_var = torch.reshape(
            x_cov_diag[..., None, :] * self.scales[:, None]**2, shape
        )
        # [..., (max_dim - min_dim) * 3 * 2]
        return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi, device=y.device)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]
 
    @property
    def out_features(self):
        return (self.max_dim - self.min_dim) * 2


class MipNeRF(nn.Module):
    def __init__(self, pos_dim=(16, 0), view_dim=4, dense_features=256, dense_depth=8) -> None:
        super().__init__()

        # positional encoding in paper
        self.pos_encoder = _IEncoder(*pos_dim)
        self.view_encoder = _Encoder(dim=view_dim)
        # net
        self.model = _NeRF(
            inpos_features=self.pos_encoder.out_features * 3,
            inview_features=self.view_encoder.out_features * 3,
            dense_features=dense_features,
            dense_depth=dense_depth,
            skips=[4]
        )

    def forward(self, x) -> torch.Tensor:
        pos_locs, view_dirs = torch.split(
            x, (6, 3), dim=-1  # locs(6) = mean(3) + var(3)
        )
        pos_locs = self.pos_encoder(pos_locs)
        view_dirs = self.view_encoder(view_dirs)
        x = torch.cat((pos_locs, view_dirs), -1)
        return self.model(x)
