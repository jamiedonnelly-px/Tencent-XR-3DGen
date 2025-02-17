from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

from model.zero123plus_unet import UNet2DConditionModelRamping

import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available

from pdb import set_trace as st

def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    device = None,
    timesteps = None,
    sigmas = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        st()
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class RefOnlyNoisedUNet(torch.nn.Module):
    # def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler) -> None:
    def __init__(self, unet:UNet2DConditionModelRamping, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]

        self.unet.set_is_controlnet(is_controlnet=False)

        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )
    
    def ramping_add(self, encoder_hidden_states_prompt, cond_encoded_clip):
        return self.unet.ramping_add(encoder_hidden_states_prompt, cond_encoded_clip)

    # def forward(
    #     self, sample, timestep, encoder_hidden_states, class_labels=None,
    #     *args, cross_attention_kwargs,
    #     down_block_res_samples=None, mid_block_res_sample=None,
    #     **kwargs
    # ):
    def forward(
        self, 
        sample, 
        timestep, 
        encoder_hidden_states_prompt=None,
        cond_encoded_clip=None,
        drop_idx=None,
        encoder_hidden_states=None, 
        class_labels=None,
        *args, 
        cross_attention_kwargs,
        down_block_res_samples=None, 
        mid_block_res_sample=None,
        **kwargs
    ):
        # print("ref running !!!")
        # breakpoint()
        if encoder_hidden_states_prompt is not None and encoder_hidden_states is None:
            encoder_hidden_states = self.ramping_add(encoder_hidden_states_prompt, cond_encoded_clip)
            encoder_hidden_states = encoder_hidden_states.to(encoder_hidden_states_prompt.dtype)
        if drop_idx is not None:
            encoder_hidden_states[drop_idx] = encoder_hidden_states_prompt[drop_idx]

        cond_lat = cross_attention_kwargs['cond_lat']
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)

        if self.training:
            # print("training !!!!")
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        else:
            # print("test !!!!")
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))
        ref_dict = {}

        # breakpoint()

        # print("noisy_cond_lat: ", noisy_cond_lat.shape)
        # print("encoder_hidden_states: ", encoder_hidden_states.shape)
        # print("timestep: ", timestep.shape)
        # breakpoint()

        self.forward_cond(
            noisy_cond_lat, timestep,
            encoder_hidden_states, class_labels,
            ref_dict, is_cfg_guidance, **kwargs
        )
        weight_dtype = self.unet.dtype
        
        # print("after forward cond running !!!")
        # breakpoint()

        self.unet.set_is_controlnet(is_controlnet=True)
        
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


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


class DepthControlUNet(torch.nn.Module):
    def __init__(self, unet: RefOnlyNoisedUNet, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, *args, cross_attention_kwargs: dict, **kwargs):
        # print("running controlnet !!!")
        # print("scale: ", self.conditioning_scale)
        # breakpoint()
        # cross_attention_kwargs = dict(cross_attention_kwargs)
        # control_depth = cross_attention_kwargs.pop('control_depth')
        control_depth = cross_attention_kwargs['control_depth']
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        # print(mid_block_res_sample.shape)
        # breakpoint()
        # self.unet.set_is_controlnet(is_controlnet=True)

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class SuperNet(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        state_dict = OrderedDict((k, state_dict[k]) for k in sorted(state_dict.keys()))
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k
            return key.split('.')[0]

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class Zero123PlusPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    # unet: UNet2DConditionModel
    unet: UNet2DConditionModelRamping
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        # unet: UNet2DConditionModel,
        unet:UNet2DConditionModelRamping,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor, 
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, 
            text_encoder=text_encoder, 
            tokenizer=tokenizer,
            unet=unet, 
            scheduler=scheduler, 
            safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare(self):
        ddpm_config = {}
        for key, value in self.scheduler.config.items():
            if key in ["set_alpha_to_one", "skip_prk_steps"]:
                continue
            ddpm_config[key] = value

        # train_sched = DDPMScheduler.from_config(self.scheduler.config)
        train_sched = DDPMScheduler.from_config(ddpm_config)
        # if isinstance(self.unet, UNet2DConditionModel):
        if isinstance(self.unet, UNet2DConditionModelRamping):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def add_controlnet(self, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0):
        # print("control 11111")
        self.prepare()
        # print("control 22222222")
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        return SuperNet(OrderedDict([('controlnet', self.unet.controlnet)]))

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image
    
    def vae_decode(self, latents_grid, hnum=4, wnum=2, imgsize=256):  # [b, 4, 32*4, 32*2]
        latent_size = imgsize // 8
        assert list(latents_grid.shape) == [
            1,
            4,
            latent_size * hnum,
            latent_size * wnum,
        ], f"latents shape should be [1, 4, {latent_size*hnum}, {latent_size*wnum}]"
        image_grid = torch.zeros([1, 3, imgsize * hnum, imgsize * wnum]).to(dtype=latents_grid.dtype)
        for hi in range(hnum):
            for wi in range(wnum):
                image_grid[
                    :, :, imgsize * hi : imgsize * (hi + 1), imgsize * wi : imgsize * (wi + 1)
                ] = self.vae.decode(
                    latents_grid[
                        :, :, latent_size * hi : latent_size * (hi + 1), latent_size * wi : latent_size * (wi + 1)
                    ]
                    / self.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
        return image_grid

    def cfg_scheduler(self, t, cfg_scheduler_type='linear', cfg_scheduler_params=[7.5,3]):
        """
        Schedular for classifier free guidance.
        cfg_scheduler_type: 'linear' or 
        cfg_scheduler_params: parameters for the scheduler
        """

        if cfg_scheduler_type == 'linear':
            assert len(cfg_scheduler_params) == 2
            start_cfg, end_cfg = cfg_scheduler_params
            # Linear interpolation to get current cfg
            weight = t/1000
            cfg = start_cfg * weight + end_cfg * (1-weight)
            return cfg

    @torch.no_grad()
    def __call_sd__(
        self,
        prompt = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps = None,
        sigmas = None,
        guidance_scale: float = 7.5,
        negative_prompt = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ["latents"],
        early_stop_noise_ratio: Optional[float] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    cfg_scheduler_type = self.cfg_scheduler_type
                    cfg_scheduler_params = self.cfg_scheduler_params
                    # cfg_scheduler_type = 'linear'
                    # cfg_scheduler_params = [2,1]
                    if cfg_scheduler_type is not None:
                        guidance_scale = self.cfg_scheduler(t, cfg_scheduler_type, cfg_scheduler_params)
                        print("[CFG SCHEDULER] guidance_scale: ", guidance_scale)
                    else:
                        guidance_scale = self.guidance_scale

                    # self.perview_cfg_weight = [1.5, 3.0, 3.5, 3.0]
                    if self.perview_cfg_weight is not None:
                        latent_h, latent_w = noise_pred_uncond.shape[2], noise_pred_uncond.shape[3]

                        if latent_h == latent_w*1.5: # 6 views
                            perview_h, perview_w = int(latent_h/3), int(latent_w/2)

                            noise_pred = noise_pred_uncond.clone()
                            noise_pred[:,:,:perview_h,:perview_w] += (guidance_scale ** self.perview_cfg_weight[0]) * (noise_pred_text[:,:,:perview_h,:perview_w] - noise_pred_uncond[:,:,:perview_h,:perview_w])
                            noise_pred[:,:,:perview_h,perview_w:] += (guidance_scale ** self.perview_cfg_weight[1]) * (noise_pred_text[:,:,:perview_h,perview_w:] - noise_pred_uncond[:,:,:perview_h,perview_w:])
                            noise_pred[:,:,perview_h:2*perview_h,:perview_w] += (guidance_scale ** self.perview_cfg_weight[2]) * (noise_pred_text[:,:,perview_h:2*perview_h,:perview_w] - noise_pred_uncond[:,:,perview_h:2*perview_h,:perview_w])
                            noise_pred[:,:,perview_h:2*perview_h,perview_w:] += (guidance_scale ** self.perview_cfg_weight[3]) * (noise_pred_text[:,:,perview_h:2*perview_h,perview_w:] - noise_pred_uncond[:,:,perview_h:2*perview_h,perview_w:])
                            noise_pred[:,:,2*perview_h:,:perview_w] += (guidance_scale ** self.perview_cfg_weight[4]) * (noise_pred_text[:,:,2*perview_h:,:perview_w] - noise_pred_uncond[:,:,2*perview_h:,:perview_w])
                            noise_pred[:,:,2*perview_h:,perview_w:] += (guidance_scale ** self.perview_cfg_weight[5]) * (noise_pred_text[:,:,2*perview_h:,perview_w:] - noise_pred_uncond[:,:,2*perview_h:,perview_w:])
                        
                            print("[CFG SCHEDULER] scaled guidance_scale (view 0): ", (guidance_scale ** self.perview_cfg_weight[0]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 1): ", (guidance_scale ** self.perview_cfg_weight[1]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 2): ", (guidance_scale ** self.perview_cfg_weight[2]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 3): ", (guidance_scale ** self.perview_cfg_weight[3]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 4): ", (guidance_scale ** self.perview_cfg_weight[4]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 5): ", (guidance_scale ** self.perview_cfg_weight[5]))
                        else:
                            perview_h, perview_w = int(latent_h/2), int(latent_w/2)

                            noise_pred = noise_pred_uncond.clone()
                            noise_pred[:,:,:perview_h,:perview_w] += (guidance_scale ** self.perview_cfg_weight[0]) * (noise_pred_text[:,:,:perview_h,:perview_w] - noise_pred_uncond[:,:,:perview_h,:perview_w])
                            noise_pred[:,:,:perview_h,perview_w:] += (guidance_scale ** self.perview_cfg_weight[1]) * (noise_pred_text[:,:,:perview_h,perview_w:] - noise_pred_uncond[:,:,:perview_h,perview_w:])
                            noise_pred[:,:,perview_h:,:perview_w] += (guidance_scale ** self.perview_cfg_weight[2]) * (noise_pred_text[:,:,perview_h:,:perview_w] - noise_pred_uncond[:,:,perview_h:,:perview_w])
                            noise_pred[:,:,perview_h:,perview_w:] += (guidance_scale ** self.perview_cfg_weight[3]) * (noise_pred_text[:,:,perview_h:,perview_w:] - noise_pred_uncond[:,:,perview_h:,perview_w:])
                        
                            print("[CFG SCHEDULER] scaled guidance_scale (view 0): ", (guidance_scale ** self.perview_cfg_weight[0]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 1): ", (guidance_scale ** self.perview_cfg_weight[1]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 2): ", (guidance_scale ** self.perview_cfg_weight[2]))
                            print("[CFG SCHEDULER] scaled guidance_scale (view 3): ", (guidance_scale ** self.perview_cfg_weight[3]))
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                step_results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                latents = step_results['prev_sample']
                pred_x0 = step_results['pred_original_sample']

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if early_stop_noise_ratio is not None:
                    if i / len(timesteps) >= (1 - early_stop_noise_ratio):
                        self._interrupt = True
                        latents = pred_x0
                        print("[D2RGB] Early stopping at step", i)
                        break


                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return ImagePipelineOutput(images=image)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt = "",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        conditioning_scale=1.5,
        depth_image: Image.Image = None,
        controlnet = None,
        output_type: Optional[str] = "pil",
        width=1024,
        height=1024,
        num_inference_steps=28,
        return_dict=True,
        early_stop_noise_ratio: Optional[float] = None,
        **kwargs
    ):
        self.prepare()
        # conditioning_scale = 1.5
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        self.unet = self.unet.to(self.vae.device, dtype=self.vae.dtype)
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)
        # print("111111")
        image = to_rgb_image(image)
        image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values
        # print("222222")
        # if depth_image is not None and hasattr(self.unet, "controlnet"):
        #     # print("depth 1111")
        #     depth_image = to_rgb_image(depth_image)
        #     depth_image = self.depth_transforms_multi(depth_image).to(
        #         device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
        #     )

        # print("depth 1111")
        depth_image = to_rgb_image(depth_image)
        depth_image = self.depth_transforms_multi(depth_image).to(
            device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
        )

        # print("33333")
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(image)
        # print("44444")
        if guidance_scale > 1:
            # print("55555")
            negative_lat = self.encode_condition_image(torch.zeros_like(image))
            cond_lat = torch.cat([negative_lat, cond_lat])
        # print("66666")
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        # print("777777")
        
        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )

        if "ramping_coefficients" in self.unet.config:
            # print("888888")
            encoder_hidden_states = self.unet.ramping_add(encoder_hidden_states, global_embeds)
        else:
            ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
            encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        # ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        # encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        # cak = dict(cond_lat=cond_lat)
        cross_attention_kwargs = {
            'cond_lat':cond_lat,
            'control_depth':depth_image
        }
        # if hasattr(self.unet, "controlnet"):
            # cak['control_depth'] = depth_image
        # cak['control_depth'] = depth_image
        # print("here 1111 !!!")
        # breakpoint()
        # from pdb import set_trace as st
        # st()

        # latents: torch.Tensor = super().__call__(
        #     None,
        #     *args,
        #     cross_attention_kwargs=cross_attention_kwargs,
        #     guidance_scale=guidance_scale,
        #     num_images_per_prompt=num_images_per_prompt,
        #     prompt_embeds=encoder_hidden_states,
        #     num_inference_steps=num_inference_steps,
        #     output_type='latent',
        #     width=width,
        #     height=height,
        #     **kwargs
        # ).images

        latents: torch.Tensor = self.__call_sd__(
            None,
            *args,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            output_type='latent',
            width=width,
            height=height,
            early_stop_noise_ratio=early_stop_noise_ratio,
            **kwargs
        ).images

        # latents = unscale_latents(latents)
        # if not output_type == "latent":
        #     # image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        #     image = unscale_image(self.vae_decode(latents, hnum=2, wnum=2, imgsize=512))
        # else:
        #     image = latents

        if not output_type == "latent":
            latents = unscale_latents(latents.to(self.vae.dtype))
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents)[0]
            image = unscale_image(image)
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)