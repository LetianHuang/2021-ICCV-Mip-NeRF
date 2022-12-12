"""
render
======
Provides
* Volume Rendering with Radiance Fields in the paper
    --- implemented by VolumeRenderer
    --- cast rays and calculate the integral to solve the disintegration rendering equation
    --- The integral is computed by Monte Carlo integral method 
        Compute the dot product of the tensors through the sampling points
* Volume Rendering with Mip-NeRF in the paper
    --- implemented by MipVolumeRenderer
* save image
    --- using OpenCV
-------------------------------------------------------------------------------------
Author: LT H
Github: mofashaoye
"""
import cv2 as cv
import numpy as np
import torch
import tqdm


def get_screen_batch(
    height: int,
    width: int,
    render_batch_size: int,
    bias: int,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """
    Get Screen Coordinates Batch
    ============================
    Inputs:
        height              : int                   scene's height
        width               : int                   scene's width
        render_batch_size   : int                   batch size of rendering
        bias                : int                   bias from [0,0]
        device              : torch.device          Output's device
    Output:
        coords              : torch.Tensor          batch coordinates of rendering
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0, height - 1, height, device=device),
            torch.linspace(0, width - 1, width, device=device)
        ),
        dim=-1
    )
    coords = torch.reshape(coords, (-1, 2))
    coords = coords[bias: bias + render_batch_size].long()
    return coords



def save_img(img: np.ndarray, path):
    """
    Save Image Using OpenCV
    """
    cv.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


def resize_img(img: np.ndarray, H, W):
    return cv.resize(img, (W, H), interpolation=cv.INTER_AREA)


class VolumeRenderer:
    """
    VolumeRenderer
    ==============
    Render the scene represented by NeRF using volume rendering
    """

    def __init__(self, nerf, width, height, focal, tnear, tfar, num_samples, num_isamples, background_w, ray_chunk, sample5d_chunk, is_train, device) -> None:
        """
        VolumeRender Constructor
        ========================
        nerf            : Neural Radiance Fields (contains encoding, coarse net and fine net)
        width           : width of the rendering scene
        height          : height of the rendering scene
        focal           : focal length
        tnear           : $t_n$ in paper
        tfar            : $t_f$ in paper
        num_samples     : number of sampling
        num_isamples    : number of Hierarchical volume sampling
        ray_chunk       : chunk of ray casting
        sample5d_chunk  : chunk of net
        background_w    : whether or not transform image's background to white
        device          : device of the whole volume renderer
        is_train        : train or eval(just rendering or test)
        """
        self.nerf = nerf                        # Neural Radiance Fields (contains encoding, coarse net and fine net)
        self.width = width                      # width of the rendering scene
        self.height = height                    # height of the rendering scene
        self.focal = focal                      # focal length
        self.tnear = tnear                      # $t_n$ in paper
        self.tfar = tfar                        # $t_f$ in paper
        self.num_samples = num_samples          # number of sampling
        self.num_isamples = num_isamples        # number of Hierarchical volume sampling
        self.ray_chunk = ray_chunk              # chunk of ray casting
        self.sample5d_chunk = sample5d_chunk    # chunk of net
        self.background_w = background_w        # whether or not transform image's background to white
        self.device = device                    # device of the whole volume renderer

        self.nerf.to(device)                    # to device(CPU or GPU)
        
        self.train(is_train)                    # train or eval(just rendering or test)

    def train(self, is_train):
        """ Train or Eval(just rendering or test) """
        self.is_train = is_train
        self.nerf.train(is_train)

    def _generate_rays(self, camera2world: torch.Tensor):
        """
        Generate Camera Rays
        ====================
        Rays Directions: 
            first generate pixel coordinates [0,W-1] x [0, H-1]
            then transform the pixel coordinates to world coordinates
            Screen Space => Camera Space => World Space
        Rays Origins: 
            get from the Camera2World matrix
        """
        # Generating pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, self.width - 1, self.width, device=self.device) + 0.5,
            torch.linspace(0, self.height - 1, self.height, device=self.device) + 0.5
        )
        i = i.t()
        j = j.t()
        # pixel coordinates to camera coordinates
        # and camera coordinates to the world coordinates
        rays_d = torch.matmul(
            torch.stack([
            (i - 0.5 * self.width) / self.focal,
            -(j - 0.5 * self.height) / self.focal,
            -torch.ones_like(i, device=self.device)
            ], dim=-1),
            camera2world[:3, :3].t()
        )
        # Camera's World Coordinate
        rays_o = camera2world[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def _sample(self, rays):
        """ 
        Volume Sampling 
        ===============
        **we partition [tn,tf ] into N evenly-spaced bins and
        then draw one sample uniformly at random from within each bin**
        Input:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
        Output:
            pos_locs: tensor [num_rays, num_samples, 3]  spatial locations used for the input of Coarse Net
            t: tensor       [num_rays, num_samples]  t of sampling 
        """
        # rays_o.shape=[num_rays, 3], rays_d.shape=[num_rays, 3]
        rays_o, rays_d = rays
        if self.is_train:
            t = torch.linspace(
                float(self.tnear), float(self.tfar), steps=self.num_samples + 1,
                device=self.device
            )
            t = t.expand((*rays_o.shape[:-1], self.num_samples + 1))
            lower, upper = t[..., :-1], t[..., 1:]
            # Linear interpolation [lower, upper] => [num_rays, num_samples]
            t = torch.lerp(lower, upper, torch.rand_like(lower))
        else:
            t = torch.linspace(
                float(self.tnear), float(self.tfar), steps=self.num_samples,
                device=self.device
            )
            t = t.expand((*rays_o.shape[:-1], self.num_samples))
        pos_locs = rays_o[..., None, :] + rays_d[..., None, :] * t[..., :, None]
        # [num_rays, num_samples, 3]

        return pos_locs, t

    def _parse_voxels(self, voxels, t_vals, rays_d) -> dict:
        """
        The volume rendering integral equation
        was calculated by Monte Carlo integral method
        Inputs:
            voxels: tensor [num_rays, num_samples, 4]   results of NN forward
            t_vals: tensor [num_rays, num_samples]      t of sampling 
            rays_d: tensor [num_rays, 3]                rays' directions
        Output:
            rbg_map and cdf_map
            rbg_map: tensor [num_rays, 3]               RGB map of the rendering scene
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        """
        t_delta = t_vals[..., 1:] - t_vals[..., :-1]
        t_delta = torch.cat(
            (t_delta, torch.tensor(
                [1e10], device=self.device).expand_as(t_delta[..., :1])),
            dim=-1
        ) * torch.norm(rays_d[..., None, :], dim=-1) # [num_rays, num_samples]
        
        c_i = torch.sigmoid(voxels[..., :3]) # [num_rays, num_samples, 3]
        alpha_i = 1 - torch.exp(-torch.relu(voxels[..., 3]) * t_delta)
        # exp(a + b) == exp(a) * exp(b)
        w_i = alpha_i * torch.cumprod(
            torch.cat(
                (torch.ones((*alpha_i.shape[:-1], 1), device=self.device), 1.0 - alpha_i + 1e-10),
                dim=-1
            ),
            dim=-1
        )[:, :-1]  
        # [:, :-1]   [num_rays, num_samples + 1] => [num_rays, num_samples]
        rgb_map = torch.sum(
            w_i[..., None] * c_i, # [num_rays, num_samples, 1] * [num_rays, num_samples, 3]
            dim=-2,  # num_samples
            keepdim=False
        )  # [num_rays, 3]
        
        # Normalizing these weights as ˆwi = wi/∑Ncj=1 wj 
        # produces a piecewise-constant PDF along the ray. **5.2**
        pdf_map = w_i[..., 1:-1] + 1e-5  
        # prevent nans
        pdf_map = pdf_map / torch.sum(pdf_map, -1, keepdim=True)
        cdf_map = torch.cumsum(pdf_map, dim=-1)
        cdf_map = torch.cat(
            (torch.zeros_like(cdf_map[..., :1]), cdf_map), dim=-1
        )

        if self.background_w:
            rgb_map = rgb_map + (1.0 - torch.sum(w_i, dim=-1, keepdim=False)[..., None])

        return dict(
            rgb_map=rgb_map,
            cdf_map=cdf_map
        )

    def _hierarchical_sample(self, rays, t_vals: torch.Tensor, cdf_map: torch.Tensor):
        """
        Hierarchical volume sampling in paper
        =====================================
        **We sample a second set of Nf locations from this distribution
        using inverse transform sampling, evaluate our “fine” network at the union of the
        first and second set of samples, and compute the final rendered color of the ray Cf (r) using Eqn. 3 but using all Nc + Nf samples.**
        Follow the official approach. 
        Inputs:
            rays: (rays_o, rays_d) tuple[tensor, tensor]
            t_vals: tensor [num_rays, num_samples] t of the result of uniform sampling (the function `_sample`)
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        Output:
            pos_locs: tensor [num_rays, num_samples + num_isamples, 3]  spatial locations used for the input of Fine Net
            t_vals: tensor [num_rays, num_samples + num_isamples] t of sampling and hierarchical sampling
        """
        rays_o, rays_d = rays
        t_vals_mid = (t_vals[..., :-1] + t_vals[..., 1:]) * 0.5 
        # [num_rays, num_samples - 1] == cdf_map.shape

        u = torch.rand((*cdf_map.shape[:-1], self.num_isamples), device=self.device).contiguous() # [num_rays, num_isamples]
        
        index = torch.searchsorted(cdf_map, u, right=True) # [num_rays, num_isamples] find > u
        index = torch.stack((
            torch.max(torch.zeros_like(index, device=self.device), index - 1), 
            torch.min(torch.full_like(index, fill_value=(cdf_map.shape[-1] - 1) * 1.0, device=self.device), index) 
            ), dim=-1) # [num_rays, num_isamples, 2]
        
        shape_m = [index.shape[0], index.shape[1], cdf_map.shape[-1]] # [num_rays, num_isamples, num_samples - 1]
        cdf_map_gather = torch.gather(cdf_map.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        t_vals_gather = torch.gather(t_vals_mid.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        
        denom = cdf_map_gather[..., 1] - cdf_map_gather[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=self.device), denom) # [num_rays, num_isamples]
        
        t_vals_fine = torch.lerp(t_vals_gather[..., 0], t_vals_gather[..., 1], (u - cdf_map_gather[..., 0]) / denom).detach()
        t_vals, _ = torch.sort(torch.cat((t_vals, t_vals_fine), dim=-1), dim=-1)

        pos_locs = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None] # [num_rays, num_samples + num_isamples, 3]

        return pos_locs, t_vals

    def _cast_rays(self, rays):
        """
        Rays Casting
        ============
        1. sample spatial locations and t for coarse net of NeRF
        2. using spatial locations and view directions as inputs of the coarse net of NeRF 
            to get voxels (rgb density) of the spatial locations
        3. parse voxels (rgb density) to rgb map of the scene (calculate integral) 
            and cdf of the sampling t
        4. hierarchical sample spatial locations and t for fine net of NeRF
        5. using spatial locations obtained by [4] and view directions as inputs of the fine net of NeRF 
            to get voxels (rgb density) of the spatial locations
        6. return the results (rgb map) of [2] and [5]
        Inputs:
            rays: (rays_o, rays_d) 
            view_dirs: tensor  view directions [num_rays, 3]
        Outputs:
            rgb map of [2] and [5]
        """
        rays_o, rays_d = rays
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        coarse_nerf_locs, t_coarse = self._sample((rays_o, rays_d))
        coarse_voxels = self._voxel_sample5d(
            torch.cat((coarse_nerf_locs,
                       view_dirs[..., None, :].expand_as(coarse_nerf_locs)), dim=-1),
            "coarse"
        )
        coarse_info = self._parse_voxels(coarse_voxels, t_coarse, rays_d)
        fine_nerf_locs, t_fine = self._hierarchical_sample(
            (rays_o, rays_d), t_coarse, coarse_info["cdf_map"]
        )
        fine_voxels = self._voxel_sample5d(
            torch.cat((fine_nerf_locs,
                       view_dirs[..., None, :].expand_as(fine_nerf_locs)), dim=-1),
            "fine"
        )
        fine_info = self._parse_voxels(fine_voxels, t_fine, rays_d)
        return coarse_info["rgb_map"], fine_info["rgb_map"]

    def _voxel_sample5d(self, x: torch.Tensor, net_type) -> torch.Tensor:
        """
        sample voxels (rgb density) using NeRF
        =======================================
        Inputs:
            x : spatial locations and view directions
            net_type: "coarse" or "fine" to select coarse net or fine net to forward
        Outputs:
            rgb + density
        """
        self.nerf.net_(net_type)
        if self.sample5d_chunk is None or self.sample5d_chunk <= 1:
            return self.nerf(x)
        return torch.cat(
            [self.nerf(x[i:i + self.sample5d_chunk])
             for i in range(0, x.shape[0], self.sample5d_chunk)],
            dim=0
        )

    def render(self, pose, select_coords=None):
        """
        render some pixels of the scene 
        (mainly used in training NeRF and `self.render_image`)
        Inputs:
            pose: camera pose (camera2world matrix)
            select_coords: some random pixels of the scene
        Outputs:
            rgb maps of coarse net and fine net 
        """
        rays_o, rays_d = self._generate_rays(pose)

        if select_coords is not None:
            rays_o = rays_o[select_coords[..., 0], select_coords[..., 1]]
            rays_d = rays_d[select_coords[..., 0], select_coords[..., 1]]

        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        
        if self.ray_chunk is None or self.ray_chunk <= 1:
            return self._cast_rays((rays_o, rays_d))

        coarse_result, fine_result = [], []

        for i in range(0, rays_o.shape[0], self.ray_chunk):
            j = i + self.ray_chunk

            c, f = self._cast_rays((rays_o[i:j], rays_d[i:j]))
            coarse_result.append(c)
            fine_result.append(f)

        return torch.cat(coarse_result, dim=0), torch.cat(fine_result, dim=0)

    def render_image(self, pose, render_batch_size=1024, use_tqdm=True) -> np.ndarray:
        """
        render a scene (image)
        ======================
        * work in eval state (just rendering not for training)
        * NumPy.NDArray => torch.Tensor(CPU) => torch.Tensor(GPU) => torch.Tensor(CPU) => NumPy.NDArray
        Inputs:
            pose                : NumPy.NDArray         camera pose (camera2world matrix)
            render_batch_size   : int                   batch size of rendering default is 1024 
            use_tqdm            : bool                  whether or not use `tqdm` module
        Outputs:
            img: NumPy.NDArray
        """
        self.train(False)
        img_block_list = []
        pose = torch.tensor(pose, device=self.device)
        if use_tqdm:
            for epoch in tqdm.trange(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(self.height, self.width, render_batch_size, epoch, device=self.device)
                _, image_fine = self.render(pose, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        else:
            for epoch in range(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(self.height, self.width, render_batch_size, epoch, device=self.device)
                _, image_fine = self.render(pose, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        img = np.concatenate(img_block_list, axis=0)[:self.height * self.width].reshape(self.height, self.width, 3)
        return img


class MipVolumeRenderer:
    """
    VolumeRenderer
    ==============
    Render the scene represented by NeRF using volume rendering
    """

    def __init__(self, mip_nerf, width, height, focal, tnear, tfar, num_samples, num_isamples, background_w, ray_chunk, sample5d_chunk, is_train, device) -> None:
        """
        VolumeRender Constructor
        ========================
        nerf            : Neural Radiance Fields (contains encoding, coarse net and fine net)
        width           : width of the rendering scene
        height          : height of the rendering scene
        focal           : focal length
        tnear           : $t_n$ in paper
        tfar            : $t_f$ in paper
        num_samples     : number of sampling
        num_isamples    : number of Hierarchical volume sampling
        ray_chunk       : chunk of ray casting
        sample5d_chunk  : chunk of net
        background_w    : whether or not transform image's background to white
        device          : device of the whole volume renderer
        is_train        : train or eval(just rendering or test)
        """
        self.mip_nerf = mip_nerf                # Neural Radiance Fields (contains encoding, coarse net and fine net)
        self.width = width                      # width of the rendering scene
        self.height = height                    # height of the rendering scene
        self.focal = focal                      # focal length
        self.tnear = tnear                      # $t_n$ in paper
        self.tfar = tfar                        # $t_f$ in paper
        self.num_samples = num_samples          # number of sampling
        self.num_isamples = num_isamples        # number of Hierarchical volume sampling
        self.ray_chunk = ray_chunk              # chunk of ray casting
        self.sample5d_chunk = sample5d_chunk    # chunk of net
        self.background_w = background_w        # whether or not transform image's background to white
        self.device = device                    # device of the whole volume renderer

        self.mip_nerf.to(device)                # to device(CPU or GPU)
        
        self.train(is_train)                    # train or eval(just rendering or test)

    
    def change_res(self, H, W, F):
        self.height = H
        self.width = W
        self.focal = F

    
    def train(self, is_train):
        """ Train or Eval(just rendering or test) """
        self.is_train = is_train
        self.mip_nerf.train(is_train)

    def _generate_rays(self, camera2world: torch.Tensor):
        """
        Generate Camera Rays
        ====================
        Rays Directions: 
            first generate pixel coordinates [0,W-1] x [0, H-1]
            then transform the pixel coordinates to world coordinates
            Screen Space => Camera Space => World Space
        Rays Origins: 
            get from the Camera2World matrix
        """
        # Generating pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, self.width - 1, self.width, device=self.device) + 0.5,
            torch.linspace(0, self.height - 1, self.height, device=self.device) + 0.5
        )
        i = i.t()
        j = j.t()
        # pixel coordinates to camera coordinates
        # and camera coordinates to the world coordinates
        rays_d = torch.matmul(
            torch.stack([
            (i - 0.5 * self.width) / self.focal,
            -(j - 0.5 * self.height) / self.focal,
            -torch.ones_like(i, device=self.device)
            ], dim=-1),
            camera2world[:3, :3].t()
        )
        # Camera's World Coordinate
        rays_o = camera2world[:3, -1].expand(rays_d.shape)
        
        rays_r = rays_d[0, 0, :] - rays_d[0, 1, :]
        rays_r = rays_r.norm()
        
        return rays_o, rays_d, rays_r


    def lift_gaussian(self, d, t_mean, t_var, r_var):
        """Lift a Gaussian defined along a ray to 3D coordinates."""
        mean = d[..., None, :] * t_mean[..., None] 
        # [num_rays, 1, 3] * [num_rays, samples, 1] == [num_rays, samples, 3]

        d_mag_sq = torch.max(
            torch.tensor(1e-10, device=self.device), 
            torch.sum(d**2, dim=-1, keepdim=True)
        )
        # [num_rays, 1]

        d_outer_diag = d**2 # [num_rays, 3]
        null_outer_diag = 1 - d_outer_diag / d_mag_sq # [num_rays, 3]
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :] 
        # [num_rays, samples, 1] * [num_rays, 1, 3] == [num_rays, samples, 3]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        # [num_rays, samples, 3]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag

    def conical_frustum_to_gaussian(self, d, t0, t1, base_radius):
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
        return self.lift_gaussian(d, t_mean, t_var, r_var)

    def _parse_tvalues(self, rays, t_vals):
        o, d, radii = rays # [num_rays, 3]
        t0, t1 = t_vals[..., :-1], t_vals[..., 1:] # [num_rays, samples]
        means, covs = self.conical_frustum_to_gaussian(d, t0, t1, radii)
        means = means + o[..., None, :]
        return torch.cat((means, covs), dim=-1)
    
    def _sample(self, rays):
        """ 
        Volume Sampling 
        ===============
        **we partition [tn,tf ] into N evenly-spaced bins and
        then draw one sample uniformly at random from within each bin**
        Input:
            rays: (rays_o, rays_d, rays_r) tuple[tensor, tensor, tensor]
        Output:
            pos_locs: tensor [num_rays, num_samples, 3]  spatial locations used for the input of Coarse Net
            t: tensor       [num_rays, num_samples]  t of sampling 
        """
        # rays_o.shape=[num_rays, 3], rays_d.shape=[num_rays, 3]
        rays_o, rays_d, rays_r = rays
        if self.is_train:
            t = torch.linspace(
                float(self.tnear), float(self.tfar), steps=self.num_samples + 1,
                device=self.device
            )
            t = t.expand((*rays_o.shape[:-1], self.num_samples + 1))
            lower, upper = t[..., :-1], t[..., 1:]
            # Linear interpolation [lower, upper] => [num_rays, num_samples]
            t = torch.lerp(lower, upper, torch.rand_like(lower))
            t = torch.cat((t, upper[..., -1:]), dim=-1) # [num_rays, num_samples + 1]
        else:
            t = torch.linspace(
                float(self.tnear), float(self.tfar), steps=self.num_samples + 1,
                device=self.device
            )
            t = t.expand((*rays_o.shape[:-1], self.num_samples + 1))

        return self._parse_tvalues(rays, t), t

    def _parse_voxels(self, voxels, t_vals, rays_d) -> dict:
        """
        The volume rendering integral equation
        was calculated by Monte Carlo integral method
        Inputs:
            voxels: tensor [num_rays, num_samples, 4]   results of NN forward
            t_vals: tensor [num_rays, num_samples + 1]      t of sampling 
            rays_d: tensor [num_rays, 3]                rays' directions
        Output:
            rbg_map and cdf_map
            rbg_map: tensor [num_rays, 3]               RGB map of the rendering scene
            cdf_map: tensor [num_rays, num_samples - 1] CDF map (Cumulative Distribution Function)
        """
        t_delta = t_vals[..., 1:] - t_vals[..., :-1] # [num_rays, num_samples]
        t_delta = t_delta * torch.norm(rays_d[..., None, :], dim=-1) # [num_rays, num_samples]
        
        c_i = torch.sigmoid(voxels[..., :3]) # [num_rays, num_samples, 3]
        alpha_i = 1 - torch.exp(-torch.relu(voxels[..., 3]) * t_delta)
        # exp(a + b) == exp(a) * exp(b)
        w_i = alpha_i * torch.cumprod(
            torch.cat(
                (torch.ones((*alpha_i.shape[:-1], 1), device=self.device), 1.0 - alpha_i + 1e-10),
                dim=-1
            ),
            dim=-1
        )[:, :-1]  
        # [:, :-1]   [num_rays, num_samples + 1] => [num_rays, num_samples]
        rgb_map = torch.sum(
            w_i[..., None] * c_i, # [num_rays, num_samples, 1] * [num_rays, num_samples, 3]
            dim=-2,  # num_samples
            keepdim=False
        )  # [num_rays, 3]
        
        # Normalizing these weights as ˆwi = wi/∑Ncj=1 wj 
        # produces a piecewise-constant PDF along the ray. **5.2**
        pdf_map = w_i[..., 1:-1]
        pdf_map = 0.5 * (torch.max(w_i[..., :-2], pdf_map) + torch.max(w_i[..., 2:], pdf_map)) + 0.01
        # prevent nans
        pdf_map = pdf_map / torch.sum(pdf_map, -1, keepdim=True)
        cdf_map = torch.cumsum(pdf_map, dim=-1)
        cdf_map = torch.cat(
            (torch.zeros_like(cdf_map[..., :1]), cdf_map, torch.ones_like(cdf_map[..., :1])), dim=-1
        )

        if self.background_w:
            rgb_map = rgb_map + (1.0 - torch.sum(w_i, dim=-1, keepdim=False)[..., None])

        return dict(
            rgb_map=rgb_map,
            cdf_map=cdf_map
        )

    def _hierarchical_sample(self, rays, t_vals: torch.Tensor, cdf_map: torch.Tensor):
        """
        Hierarchical volume sampling in paper
        =====================================
        **We sample a second set of Nf locations from this distribution
        using inverse transform sampling, evaluate our “fine” network at the union of the
        first and second set of samples, and compute the final rendered color of the ray Cf (r) using Eqn. 3 but using all Nc + Nf samples.**
        Follow the official approach. 
        Inputs:
            rays: (rays_o, rays_d, rays_r) tuple[tensor, tensor, tensor]
            t_vals: tensor [num_rays, num_samples + 1] t of the result of uniform sampling (the function `_sample`)
            cdf_map: tensor [num_rays, num_samples] CDF map (Cumulative Distribution Function)
        Output:
            pos_locs: tensor [num_rays, num_samples + num_isamples, 3]  spatial locations used for the input of Fine Net
            t_vals: tensor [num_rays, num_samples + num_isamples] t of sampling and hierarchical sampling
        """
        t_vals_mid = (t_vals[..., :-1] + t_vals[..., 1:]) * 0.5 # [num_rays, num_samples]
        # [num_rays, num_samples - 1] == cdf_map.shape

        u = torch.rand((*cdf_map.shape[:-1], self.num_isamples + 1), device=self.device).contiguous() # [num_rays, num_isamples]
        
        index = torch.searchsorted(cdf_map, u, right=True) # [num_rays, num_isamples] find > u
        index = torch.stack((
            torch.max(torch.zeros_like(index, device=self.device), index - 1), 
            torch.min(torch.full_like(index, fill_value=(cdf_map.shape[-1] - 1) * 1.0, device=self.device), index) 
            ), dim=-1) # [num_rays, num_isamples, 2]
        
        shape_m = [index.shape[0], index.shape[1], cdf_map.shape[-1]] # [num_rays, num_isamples, num_samples - 1]
        cdf_map_gather = torch.gather(cdf_map.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        t_vals_gather = torch.gather(t_vals_mid.unsqueeze(1).expand(shape_m), dim=2, index=index) # [num_rays, num_isamples, 2]
        
        denom = cdf_map_gather[..., 1] - cdf_map_gather[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=self.device), denom) # [num_rays, num_isamples]
        
        t_vals_fine = torch.lerp(t_vals_gather[..., 0], t_vals_gather[..., 1], (u - cdf_map_gather[..., 0]) / denom).detach()
        t, _ = torch.sort(t_vals_fine, dim=-1)
        # t_vals, _ = torch.sort(torch.cat((t_vals, t_vals_fine), dim=-1), dim=-1)

        return self._parse_tvalues(rays, t), t

    def _cast_rays(self, rays):
        """
        Rays Casting
        ============
        1. sample spatial locations and t for coarse net of NeRF
        2. using spatial locations and view directions as inputs of the coarse net of NeRF 
            to get voxels (rgb density) of the spatial locations
        3. parse voxels (rgb density) to rgb map of the scene (calculate integral) 
            and cdf of the sampling t
        4. hierarchical sample spatial locations and t for fine net of NeRF
        5. using spatial locations obtained by [4] and view directions as inputs of the fine net of NeRF 
            to get voxels (rgb density) of the spatial locations
        6. return the results (rgb map) of [2] and [5]
        Inputs:
            rays: (rays_o, rays_d) 
            view_dirs: tensor  view directions [num_rays, 3]
        Outputs:
            rgb map of [2] and [5]
        """
        rays_o, rays_d, rays_r = rays
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        coarse_nerf_locs, t_coarse = self._sample((rays_o, rays_d, rays_r))
        coarse_voxels = self._voxel_sample5d(
            torch.cat((coarse_nerf_locs,
                       view_dirs[..., None, :].expand(*coarse_nerf_locs.shape[:-1], 3)), dim=-1),
            "coarse"
        )
        coarse_info = self._parse_voxels(coarse_voxels, t_coarse, rays_d)
        fine_nerf_locs, t_fine = self._hierarchical_sample(
            (rays_o, rays_d, rays_r), t_coarse, coarse_info["cdf_map"]
        )
        fine_voxels = self._voxel_sample5d(
            torch.cat((fine_nerf_locs,
                       view_dirs[..., None, :].expand(*fine_nerf_locs.shape[:-1], 3)), dim=-1),
            "fine"
        )
        fine_info = self._parse_voxels(fine_voxels, t_fine, rays_d)
        return coarse_info["rgb_map"], fine_info["rgb_map"]

    def _voxel_sample5d(self, x: torch.Tensor, net_type) -> torch.Tensor:
        """
        sample voxels (rgb density) using NeRF
        =======================================
        Inputs:
            x : spatial locations and view directions
            net_type: "coarse" or "fine" to select coarse net or fine net to forward
        Outputs:
            rgb + density
        """
        # self.mip_nerf.net_(net_type)
        if self.sample5d_chunk is None or self.sample5d_chunk <= 1:
            return self.mip_nerf(x)
        return torch.cat(
            [self.mip_nerf(x[i:i + self.sample5d_chunk])
             for i in range(0, x.shape[0], self.sample5d_chunk)],
            dim=0
        )

    def render(self, pose, select_coords=None):
        """
        render some pixels of the scene 
        (mainly used in training NeRF and `self.render_image`)
        Inputs:
            pose: camera pose (camera2world matrix)
            select_coords: some random pixels of the scene
        Outputs:
            rgb maps of coarse net and fine net 
        """
        rays_o, rays_d, rays_r = self._generate_rays(pose)

        if select_coords is not None:
            rays_o = rays_o[select_coords[..., 0], select_coords[..., 1]]
            rays_d = rays_d[select_coords[..., 0], select_coords[..., 1]]

        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        
        if self.ray_chunk is None or self.ray_chunk <= 1:
            return self._cast_rays((rays_o, rays_d, rays_r))

        coarse_result, fine_result = [], []

        for i in range(0, rays_o.shape[0], self.ray_chunk):
            j = i + self.ray_chunk

            c, f = self._cast_rays((rays_o[i:j], rays_d[i:j], rays_r))
            coarse_result.append(c)
            fine_result.append(f)

        return torch.cat(coarse_result, dim=0), torch.cat(fine_result, dim=0)

    def render_image(self, pose, render_batch_size=1024, use_tqdm=True) -> np.ndarray:
        """
        render a scene (image)
        ======================
        * work in eval state (just rendering not for training)
        * NumPy.NDArray => torch.Tensor(CPU) => torch.Tensor(GPU) => torch.Tensor(CPU) => NumPy.NDArray
        Inputs:
            pose                : NumPy.NDArray         camera pose (camera2world matrix)
            render_batch_size   : int                   batch size of rendering default is 1024 
            use_tqdm            : bool                  whether or not use `tqdm` module
        Outputs:
            img: NumPy.NDArray
        """
        self.train(False)
        img_block_list = []
        pose = torch.tensor(pose, device=self.device)
        if use_tqdm:
            for epoch in tqdm.trange(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(self.height, self.width, render_batch_size, epoch, device=self.device)
                _, image_fine = self.render(pose, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        else:
            for epoch in range(0, self.height * self.width, render_batch_size):
                coords = get_screen_batch(self.height, self.width, render_batch_size, epoch, device=self.device)
                _, image_fine = self.render(pose, select_coords=coords)
                img_block_list.append(image_fine.detach().cpu())
        img = np.concatenate(img_block_list, axis=0)[:self.height * self.width].reshape(self.height, self.width, 3)
        return img