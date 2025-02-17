# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from math import sin, cos
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

# Manifold constrain
from torch.autograd import grad

from pdb import set_trace as st
from matplotlib import pyplot as plt

# Baking
# import os, sys
# sys.path.append("/aigc_cfs_2/jiayuuyang/texture_refine/src/")
# from nv_diff_bake import NvdiffRender

# New Baking
import os, sys
# sys.path.append("/aigc_cfs_2/zacheng/demo_render/render_bake/")
# from render_bake_utils_v2_2stage_90s import dilate_masks, Renderer
from render_bake_utils_v5_pbr import dilate_masks, Renderer
import cv2, torch, numpy as np

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerAncestralDiscrete
class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


class ConsistentEulerAncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Ancestral sampling with Euler method steps.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def init_renderer(self, 
        mesh=None, 
        render_size=512, 
        scale_factor=1.0, 
        renderer_extra_scale_up_factor=1.0, 
        d2rgb_special_scale=False, 
        texture_resolution=2048, 
        init_texture=None, 
        num_inference_steps=None,
        apply_consistent_interval=0,
        upsample_final_views=False,
        rotate_input_mesh=False,
        ctx=None,
        unwrap_uv=False,
    ):
        
        # New baking
        texture_resolution = texture_resolution
        obj_path = mesh
        obj_bound = scale_factor

        # # Xibin's 6 views settings
        # cam_azimuths = [0, 90, 180, 270, 0, 0] # len is n_views
        # cam_elevations = [0,0,0,0, -89.9, 89.9] # len is n_views
        # cam_distances = [5,5,5,5,5,5] # len is n_views
        # camera_type = "ortho"

        cam_azimuths = [0, 90, 180, 270] # len is n_views
        cam_elevations = [0,0,0,0] # len is n_views
        cam_distances = [5,5,5,5] # len is n_views
        camera_type = "ortho"

        bake_weight_exp = 3.0
        bake_front_view_weight = 10.0
        bake_erode_boundary = 10

        image_resolution = render_size

        # set up renderer
        renderer = Renderer(image_resolution, texture_resolution, world_orientation="y-up",ctx=ctx)
        renderer.set_object(obj_path, bound=obj_bound, orientation="y-up")
        self.input_has_uv = renderer.mesh.has_uv()
        if (self.input_has_uv==False) or unwrap_uv:
            unwrap_uv_success = renderer.unwrap_uv()
            if (unwrap_uv_success == False) and (self.input_has_uv == False):
                raise Exception("Failed to unwrap uv and no uv provided from input. job failed.")
        if rotate_input_mesh:
            deg = 30
            rad = math.radians(deg)
            transform = np.array([
                [cos(rad),  0,   sin(rad),  0],
                [0,         1,   0,         0],
                [-sin(rad), 0,   cos(rad),  0],
                [0,         0,          0,  1]
            ])
            renderer.transform_obj(transform)
        renderer.set_cameras(azimuths=cam_azimuths, elevations=cam_elevations, dists=cam_distances, camera_type=camera_type, zooms=1.0, near=1e-1, far=1e1)

        # render normal, depth, xyz
        depth, mask = renderer.render_depth("absolute", normalize=(255,50), bg=0) # (n_views, img_res, img_res, 1)
        normal, mask = renderer.render_normal("camera") # (n_views, img_res, img_res, 3)
        view_cos = -normal[...,-1:] # (n_views, img_res, img_res, 3)
        xyz, _ = renderer.render_xyz(system="world", antialias=True, ssaa=True, cameras=None)

        # detect depth discontinuities, i.e. occlusion boundaries
        depth_map_uint8 = depth.cpu().numpy().astype(np.uint8) # (n_views, img_res, img_res, 1)
        depth_edge = [(cv2.Canny(d, 10, 40) > 0) for d in depth_map_uint8]
        depth_edge = dilate_masks(*depth_edge, iterations=bake_erode_boundary)
        depth_edge = (torch.from_numpy(depth_edge).cuda() > 0).float().unsqueeze(-1) # binary (n_views, img_res, img_res, 1)

        weights = view_cos * (1-depth_edge) # remove pixels on occlusion boundaries
        # apply weights
        weights = weights ** bake_weight_exp
        weights[0] *= bake_front_view_weight

        self.renderer = renderer
        self.depth = depth
        self.normal = normal
        self.xyz = xyz
        self.mask = mask
        self.weights = weights
        self.renderer_extra_scale_up_factor = renderer_extra_scale_up_factor
        self.d2rgb_special_scale = d2rgb_special_scale

        self.apply_consistent_interval = apply_consistent_interval
        self.upsample_final_views = upsample_final_views

        # Render rgb initial views if texture map is provided.
        if init_texture is not None:
            # Interpoate texture map to desired resolution.
            init_texture = torch.nn.functional.interpolate(init_texture, size=(texture_resolution,texture_resolution), mode='nearest', align_corners=False)

            # render
            color, alpha = self.renderer.sample_texture(init_texture, max_mip_level=4)
            self.init_rgb_views = torch.clamp(torch.cat([color,alpha],dim=-1).permute(0,3,1,2),0,1)

        print("Renderer initialized.")

        return


    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.sigmas.max()

        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[
                ::-1
            ].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            timesteps -= 1
        elif self.config.timestep_spacing == "biased_end":
            print("[CONSISTENT SCHEDULER] Using biased_end timesteps scheduling!")
            t = np.linspace(1, 0, num_inference_steps, dtype=np.float32)
            timesteps = (t**2) * (self.config.num_train_timesteps - 1)  # Quadratic scaling
            timesteps = timesteps.round()  # Reverse to start from max and approach 0
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def split_image(self, image, rows, cols):
        width, height = image.size
        block_width = width // cols
        block_height = height // rows

        images = []
        for i in range(rows):
            for j in range(cols):
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height
                sub_image = image.crop((left, upper, right, lower))
                images.append(sub_image)

        return images

    def split_image_tensor(self, image, rows, cols):
        B, C, width, height = image.shape
        assert B == 1
        block_width = width // cols
        block_height = height // rows

        images = []
        for i in range(rows):
            for j in range(cols):
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height
                sub_image = image[0, :,left:right,upper:lower]
                images.append(sub_image)

        return images

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """
        vae = self.vae

        # Print cuda memory usage
        # device = torch.device('cuda:0')
        # free, total = torch.cuda.mem_get_info(device)
        # mem_total_mb = total / 1024 ** 2
        # mem_free_mb = free / 1024 ** 2
        # mem_used_mb = (total - free) / 1024 ** 2
        # print(f"[CONSISTENT_SCHEDULER][Step] Memory total: ",mem_total_mb)
        # print(f"[CONSISTENT_SCHEDULER][Step] Memory free: ",mem_free_mb)
        # print(f"[CONSISTENT_SCHEDULER][Step] Memory used: ",mem_used_mb)

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5


        # Save predicted original sample for debugging
        # vae_dtype = vae.dtype
        # vae = vae.to(torch.float32)
        # plt.imsave(f"outputs/tmp/pred_x0_{self._step_index}.png",(torch.clamp(vae.decode(pred_original_sample / vae.config.scaling_factor)[0][0],-1,1).permute(1,2,0).cpu().numpy()+1)/2)
        # vae = vae.to(vae_dtype)

        ##### Enforce Consistency #####
        save_intermediate_results = False
        disable_rgb_clamp = True
        apply_consistency = (self.num_inference_steps-self._step_index-1) % (self.apply_consistent_interval+1) == 0
        if self.disable_consistency:
            apply_consistency = False
        # for ii in range(10): print("[DEBUG] SDXL consistency disabled!!!\n\n")
        if apply_consistency:
            print("Apply consistency on step: ", self._step_index)
            ### Decode pred_x0 ###
            vae_dtype = vae.dtype
            # vae = vae.to(torch.float32)
            pred_original_sample_dtype = pred_original_sample.dtype
            if self.d2rgb_special_scale:
                pred_original_sample = unscale_latents(pred_original_sample.to(vae.dtype))
                pred_original_sample = pred_original_sample / vae.config.scaling_factor
                pred_rgb_x0 = vae.decode(pred_original_sample)[0]
                pred_rgb_x0 = unscale_image(pred_rgb_x0)
            else:
                pred_rgb_x0 = vae.decode(pred_original_sample.to(vae.dtype) / vae.config.scaling_factor)[0]
            if save_intermediate_results:
                pred_rgb_x0_path = os.path.join(self.output_path,f"{self._step_index}_step1_pred_x0_rgb.png")
                plt.imsave(pred_rgb_x0_path,(torch.clamp(pred_rgb_x0[0],-1,1).permute(1,2,0).cpu().numpy()+1)/2)

            # Upscale for antialiasing
            diffusion_img_size = pred_rgb_x0.shape[-1]
            if self.upsample_final_views and self._step_index == self.num_inference_steps-1:
                print("[REAL-ESRGAN] Upsample final views...")
                # TODO: Avoid moving to CPU
                plt.imsave(os.path.join(self.output_path,f"final_images_before_upsample.png"),(((torch.clamp(pred_rgb_x0[0].permute(1,2,0),-1,1).float().data.cpu().numpy()+1)/2)*255).astype(np.uint8))
                pred_rgb_x0_up, _ = self.upsampler.enhance((((torch.clamp(pred_rgb_x0[0].permute(1,2,0),-1,1).float().data.cpu().numpy()+1)/2)*255).astype(np.uint8), outscale=2)
                # pred_rgb_x0_up, _ = self.upsampler_small.enhance((((torch.clamp(pred_rgb_x0[0].permute(1,2,0),-1,1).float().data.cpu().numpy()+1)/2)*255).astype(np.uint8), outscale=4)
                pred_rgb_x0_up, _ = self.upsampler_small.enhance(pred_rgb_x0_up, outscale=4)
                # pred_rgb_x0_up, _ = self.upsampler.enhance(pred_rgb_x0_up, outscale=2)
                up_size = (int(pred_rgb_x0.shape[2]*self.renderer_extra_scale_up_factor), int(pred_rgb_x0.shape[3]*self.renderer_extra_scale_up_factor))
                pred_rgb_x0 = torch.nn.functional.interpolate(torch.tensor(pred_rgb_x0_up).permute(2,0,1).unsqueeze(0), size=up_size, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
                pred_rgb_x0 = pred_rgb_x0.to(vae_dtype).to(vae.device)
                pred_rgb_x0 = ((pred_rgb_x0/255)*2)-1
            else:
                pred_rgb_x0 = torch.nn.functional.interpolate(pred_rgb_x0, size=None, scale_factor=self.renderer_extra_scale_up_factor, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)

            ### Bake to mesh ###
            from diffusers.utils import load_image, make_image_grid, numpy_to_pil

            # image_all = Image.fromarray((((torch.clamp(pred_rgb_x0[0].to(torch.float32),-1,1).permute(1,2,0).cpu().numpy()+1)/2)*255).astype(np.uint8))
            # in_images = ((torch.clamp(pred_rgb_x0[0].to(torch.float32),-1,1).permute(1,2,0)+1)/2) # [b, h, w, 3] in [0, 1]
            image_list = self.split_image_tensor(pred_rgb_x0, 2, 2)
            in_images = torch.stack([image_list[0],image_list[2],image_list[1],image_list[3],],dim=0)
            if disable_rgb_clamp:
                in_images = (in_images.to(torch.float32).permute(0,2,3,1)+1)/2
            else:
                in_images = (torch.clamp(in_images.to(torch.float32),-1,1).permute(0,2,3,1)+1)/2
            # in_images = in_images.to(self.nv_render.device)

            # if self._step_index < self.num_inference_steps-1:
            #     out_dir = None
            # else: 
            #     out_dir = os.path.join(self.output_path,f"{self._step_index}")

            # cos_exp = 5
            # with torch.enable_grad():
            #     tex_data = self.nv_render.bake_views(in_images,
            #                         out_dir,
            #                         max_mip_level=self.max_mip_level,
            #                         main_views=[0],
            #                         main_weight=100,
            #                         cos_exp=cos_exp,
            #                         save_debug=False)
            
            # New baking
            # bake
            image_weights = torch.cat((in_images, self.weights), dim=-1)
            # texture_weights = self.renderer.bake_textures_raycast(image_weights, interpolation="nearest", inpaint=False)
            texture_weights = self.renderer.bake_textures(image_weights, antialias=False, ssaa=False, inpaint=False)
            # texture_weights = self.renderer.bake_textures(image_weights, inpaint=False, max_mpimap_level=None)
            textures, weights = torch.split(texture_weights, (3,1), dim=-1)
            # blend textures by weights
            total_weights = torch.sum(weights, dim=0, keepdim=True) # (1, img_res, img_res, 1)
            texture = torch.sum(textures*weights, dim=0, keepdim=True) / (total_weights + 1e-10) # (1, img_res, img_res, 3)
            # inpaint missing regions, optional
            texture = self.renderer.inpaint_textures(texture, (total_weights<=1e-5), inpaint_method="laplace") # (1, img_res, img_res, 3)

            # Save texture map
            if save_intermediate_results or (self._step_index == self.num_inference_steps-1):
                plt.imsave(os.path.join(self.output_path,f"{self._step_index}_texture.png"),torch.clamp(texture[0],0,1).data.cpu().numpy())

            ### Render ###
            # color, alpha = self.nv_render.sample_texture(tex_data[None])
            # re-render image from textures
            color, alpha = self.renderer.sample_texture(texture, max_mip_level=4)

            # plt.imsave("debug.png",torch.clamp(color[0],0,1).data.cpu().numpy())
            if disable_rgb_clamp:
                textured_views_rgb = torch.cat([color,alpha],dim=-1).permute(0,3,1,2)
            else:
                textured_views_rgb = torch.clamp(torch.cat([color,alpha],dim=-1).permute(0,3,1,2),0,1) # range [0,1]

            # Encode rendered views into latents and use as new pred_x0
            render_size = textured_views_rgb.shape[-1]
            new_pred_x0_rgb = torch.zeros((1,3,render_size*2,render_size*2),device="cuda",dtype=torch.float32)
            # new_pred_x0_rgb[:,:,:render_size,:render_size] = textured_views_rgb[0][:3]*textured_views_rgb[0][-1:] + ((1-textured_views_rgb[0][-1:])*0.5)# Grey background
            # new_pred_x0_rgb[:,:,:render_size,render_size:] = textured_views_rgb[2][:3]*textured_views_rgb[2][-1:] + ((1-textured_views_rgb[2][-1:])*0.5)
            # new_pred_x0_rgb[:,:,render_size:,:render_size] = textured_views_rgb[1][:3]*textured_views_rgb[1][-1:] + ((1-textured_views_rgb[1][-1:])*0.5)
            # new_pred_x0_rgb[:,:,render_size:,render_size:] = textured_views_rgb[3][:3]*textured_views_rgb[3][-1:] + ((1-textured_views_rgb[3][-1:])*0.5)

            new_pred_x0_rgb[:,:,:render_size,:render_size] = textured_views_rgb[0][:3]*textured_views_rgb[0][-1:] + ((1-textured_views_rgb[0][-1:])*0.5)# Grey background
            new_pred_x0_rgb[:,:,:render_size,render_size:] = textured_views_rgb[1][:3]*textured_views_rgb[1][-1:] + ((1-textured_views_rgb[1][-1:])*0.5)
            new_pred_x0_rgb[:,:,render_size:,:render_size] = textured_views_rgb[2][:3]*textured_views_rgb[2][-1:] + ((1-textured_views_rgb[2][-1:])*0.5)
            new_pred_x0_rgb[:,:,render_size:,render_size:] = textured_views_rgb[3][:3]*textured_views_rgb[3][-1:] + ((1-textured_views_rgb[3][-1:])*0.5)

            # new_pred_x0_rgb[:,:,:render_size,:render_size] = textured_views_rgb[0][:3]*textured_views_rgb[0][-1:] + ((1-textured_views_rgb[0][-1:])*(pred_rgb_x0[0]+1)/2)# Original background
            # new_pred_x0_rgb[:,:,:render_size,render_size:] = textured_views_rgb[2][:3]*textured_views_rgb[2][-1:] + ((1-textured_views_rgb[2][-1:])*(pred_rgb_x0[2]+1)/2)
            # new_pred_x0_rgb[:,:,render_size:,:render_size] = textured_views_rgb[1][:3]*textured_views_rgb[1][-1:] + ((1-textured_views_rgb[1][-1:])*(pred_rgb_x0[1]+1)/2)
            # new_pred_x0_rgb[:,:,render_size:,render_size:] = textured_views_rgb[3][:3]*textured_views_rgb[3][-1:] + ((1-textured_views_rgb[3][-1:])*(pred_rgb_x0[3]+1)/2)

            if save_intermediate_results:
                plt.imsave(os.path.join(self.output_path,f"{self._step_index}_step2_pred_x0_rgb_baked.png"),(torch.clamp(new_pred_x0_rgb[0].to(torch.float32),0,1).permute(1,2,0).contiguous().cpu().numpy()))
            # plt.imsave("debug.png",(torch.clamp(new_pred_x0_rgb[0].to(torch.float32),0,1).permute(1,2,0).contiguous().cpu().numpy()))
            # plt.imsave("debug2.png",textured_views_rgb[2][:3].permute(1,2,0).float().data.cpu().numpy())

            # Downsample back to original size of diffusion model
            new_pred_x0_rgb = torch.nn.functional.interpolate(new_pred_x0_rgb, size=None, scale_factor=1/self.renderer_extra_scale_up_factor, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)

            # Attempt: Add gaussian noise to baked image before vae encode.
            # print("[DEBUG] sigma_up:",sigma_up)
            # new_pred_x0_rgb = new_pred_x0_rgb + sigma_up*(torch.randn(new_pred_x0_rgb.shape,device=new_pred_x0_rgb.device,dtype=new_pred_x0_rgb.dtype))/2
            # if save_intermediate_results:
            #     plt.imsave(os.path.join(self.output_path,f"{self._step_index}_step2_pred_x0_rgb_noised.png"),(torch.clamp(new_pred_x0_rgb[0].to(torch.float32),0,1).permute(1,2,0).contiguous().cpu().numpy()))

            new_pred_x0_rgb = new_pred_x0_rgb.to(vae.dtype)
            if self.d2rgb_special_scale:
                new_pred_x0_rgb = (new_pred_x0_rgb*2)-1
                new_pred_x0_rgb = scale_image(new_pred_x0_rgb)
                rendered_x0 = vae.encode(new_pred_x0_rgb).latent_dist.mode()
                rendered_x0 = rendered_x0 * vae.config.scaling_factor
                rendered_x0 = scale_latents(rendered_x0)
            else:
                rendered_x0 = vae.encode((new_pred_x0_rgb*2)-1).latent_dist.mode() * vae.config.scaling_factor

            # Use as new pred_x0
            # if self._step_index < self.num_inference_steps-1:
            pred_original_sample = rendered_x0.to(pred_original_sample_dtype)

        # print("consistent pred_x0 DISABLED!!!")
        # vae = vae.to(vae_dtype)
        # st()
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)

        prev_sample = prev_sample + noise * sigma_up

        # Override output on last step of consistent scheduler
        if apply_consistency:
            if self._step_index == self.num_inference_steps-1:
                print("Override scheduler output to be baked latent!")
                prev_sample = pred_original_sample

                # Debug
                # pred_original_sample = unscale_latents(pred_original_sample.to(vae.dtype))
                # pred_original_sample = pred_original_sample / vae.config.scaling_factor
                # pred_rgb_x0 = vae.decode(pred_original_sample)[0]
                # pred_rgb_x0 = unscale_image(pred_rgb_x0)
                # plt.imsave("debug.png",(torch.clamp(pred_rgb_x0[0],-1,1).permute(1,2,0).float().data.cpu().numpy()+1)/2)
                # st()

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps