try:
    from torchvision.transforms import v2
except:
    from torchvision import transforms as v2
    
import torch 
import random

import torch
import torch.nn.functional as F
from .image_process_utils import adjust_hsv, adjust_contrast, adjust_gauss_blur, adjust_gamma

def gen_uniform(shape, min, max):
    return torch.rand(shape, device="cpu") * (max - min) + min

def adjust(images, exposure=None, gamma=None, brightness=None, saturation=None, hue=None, contrast=None, blur=None, sigma=None, bg_noise=None, bg_color=None, **kwargs):
    """
    Adjust the color properties and sharpness of a batch of images in PyTorch.

    Parameters:
    - images: Tensor of shape [batch, 3, H, W] with pixel values in range [0,1].    
    - exposure: Tensor of shape [batch, 1] or [1,1] for broadcasting, in range [0,inf). exposure adjustment, 1 means no adjustment and 2 means double exposure.
    - gamma: Tensor of shape [batch, 1] or [1,1] for gamma adjustment, in range [0,inf). gamma<1 means better dynamic range for shadows and gamma>1 better highlights
    - brightness: Tensor of shape [batch, 1] or [1,1] for broadcasting, in range [0,inf). Brightness adjustment in HSV space, 1 means no adjustment and 2 means double brighntess.
    - saturation: Tensor of shape [batch, 1] or [1,1] for saturation adjustment, in range [0,inf).
    - hue: Tensor of shape [batch, 1] or [1,1] for hue adjustment, in range [-0.5,0.5). 0 means no adjustment.
    - contrast: Tensor of shape [batch, 1] or [1,1] for contrast adjustment, in range [0,inf).
    - blur: Tensor of shape [batch, 1] or [1,1] for blur adjustment, in range [0,1].
    - sigmas: Tensor of shape [batch, 1] or [1,1] for Gaussian blur sigma values, relative to image dimensions.
    - bg_noise: Tensor of shape [batch, 3, H, W]

    Returns:
    - Adjusted images as a Tensor of shape [batch, channel, H, W].
    """
    
    adjusted = False
    
    batch, channel, H, W = images.size()

    if exposure is not None or gamma is not None:
        if exposure is None:
            exposure = torch.ones((1, 1), device=images.device)
        if gamma is None:
            gamma = torch.ones((1, 1), device=images.device)
        
        images[:,:3] = adjust_gamma(images[:,:3], gamma, exposure)
        adjusted = True


    if brightness is not None or contrast is not None or saturation is not None:
        if brightness is None:
            brightness = torch.ones((1, 1), device=images.device)
        if saturation is None:
            saturation = torch.ones((1, 1), device=images.device)
        if hue is None:
            hue = torch.zeros((1, 1), device=images.device)
        
        images[:,:3] = adjust_hsv(images[:,:3], brightness, saturation, hue)
        adjusted = True

    if contrast is not None:
        images[:,:3] = adjust_contrast(images[:,:3], contrast)
        adjusted = True
    
    if blur is not None or sigma is not None:
        if blur is None:
            blur = torch.ones((1, 1), device=images.device)
        if sigma is None:
            sigma = 0.5 + torch.zeros((1, 1), device=images.device)
        
        images = adjust_gauss_blur(images, blur, sigma)
        adjusted = True
    
    if bg_color is not None:
        images[:,:3] = images[:,:3]*images[:, 3:] + (1-images[:, 3:]) * bg_color.unsqueeze(dim=-1).unsqueeze(dim=-1)
        
    
    if bg_noise is not None:
        images[:, :3] = images[:, :3] + (1-images[:, 3:]) * torch.nn.functional.interpolate(bg_noise, size=(H,W), mode='bilinear')

    return images, adjusted


def transform(images, translate=None, scale=None, elastic=None, **kwargs):
    """
    Apply optional translation, scaling, and elastic transformation to a batch of images in that order.
    Transformed images are padded with its border values.

    Parameters:
    - images: Tensor of shape [batch, channel, H, W], the batch of images to transform.
    - translate: Optional; Tensor of shape [batch, 2] or [1, 2] for broadcasting. The translations should be in the range [-1, 1], representing the fraction of translation relative to the image dimensions.
    - scale: Optional; Tensor of shape [batch, 1] or [1, 1] for broadcasting. The scale factors; a value greater than 1 means zooming in (making objects larger and cropping), less than 1 means zooming out (fitting more into the view).
    - elastic: Optional; Tensor of shape [batch, 2, res, res] or [1, 2, res, res] for broadcasting. The flow fields for elastic transformations, specifying the backward flow from the output image to the input image. Will be resized to image dimension [H,W].

    Returns:
    - Transformed images of shape [batch, channel, H, W], where each transformation has been applied considering the backward warping flow, ensuring correct sampling and transformation of the image data.
    """
    batch, channel, H, W = images.size()

    # Initialize identical mapping grid
    x = torch.linspace(-1, 1, W, device=images.device)
    y = torch.linspace(-1, 1, H, device=images.device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # Note: yy corresponds to H dimension and xx to W dimension
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # Shape: [batch/1, H, W, 2]
    transformed = False
    
    if elastic is not None:

        elastic = torch.nn.functional.interpolate(elastic, size=(H,W), mode='bilinear')  # [batchsize, 2, H, W]
        elastic = elastic.permute(0, 2, 3, 1) # [batchsize, h, w, 2]
        grid = grid + elastic  # Apply backward flow for elastic deformation
        transformed = True

    if scale is not None or translate is not None:
        if translate is None:
            translate = torch.zeros((1, 2), device=images.device)
        if scale is None:
            scale = torch.ones((1, 1), device=images.device)

        translate = translate.view(-1, 1, 1, 2)
        scale = scale.view(-1, 1, 1, 1)

        grid = grid / scale  
        grid = grid - translate 
        transformed = True
    
    if transformed:
        grid = torch.clamp(grid, -1, 1) # clip to [-1, 1] to preserve image borders
        images = F.grid_sample(images, grid.expand(batch,H,W,2), mode='bilinear', padding_mode='border', align_corners=True)

    return images, transformed


def gen_elastic(magnitude, smoothness, res=64, batchsize=1):
    """
    Generates an elastic displacement map for elastic transformations.

    :param resolution: intrinsic resolution of displacement map.
    :param magnitude: max magnitude of displacement as a ratio relative to the image size,
                      directly influencing the displacement scale.
    :param smoothness: Smoothness of the displacement, relative to the image size.
    :param res: internal resolution used to generate raw flow map, does not change output flow map size
    :return: A displacement map of shape [1, H, W, 2].
    """
    
    sigma = smoothness * res

    # Initialize displacement maps with random values
    dx = torch.rand([batchsize, 1, res, res], device='cpu') * 2 - 1
    dy = torch.rand([batchsize, 1, res, res], device='cpu') * 2 - 1

    # Apply Gaussian blur to smooth the displacement maps, if sigma > 0
    for d in [dx, dy]:
        if sigma > 0.3:
            kernel_size = int(4 * sigma) * 2 + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            d.copy_(v2.functional.gaussian_blur(d, kernel_size, sigma))

    # Scale displacement maps by magnitude, adjusted relative to the image size
    dx *= magnitude
    dy *= magnitude

    # Combine dx and dy into a single displacement map and permute to match expected shape
    displacement = torch.cat([dx, dy], dim=1) # [batchsize, 2, res, res]

    return displacement

def gen_transform_parameters(batch_size, scale_adjustment=None, translate_adjustment=None, elastic_deform=None, exposure_adjustment=None, gamma_adjustment=None, brightness_adjustment=None,  saturation_adjustment=None, hue_adjustment=None, contrast_adjustment=None, blur_adjustment=None, background_noise=None, background_color=None, **kwargs):

    params = dict()

    if scale_adjustment is not None: 
        scale = gen_uniform([batch_size, 1], *scale_adjustment)
        params['scale'] = scale
    
    if translate_adjustment is not None: 
        translate = gen_uniform([batch_size, 2], *translate_adjustment)
        params['translate'] = translate

    if elastic_deform is not None:
        magnitude = gen_uniform(1, *elastic_deform['magnitude']).item()
        elastic = gen_elastic(magnitude, elastic_deform['smoothness'], elastic_deform['resolution'], batch_size)
        params['elastic'] = elastic

    if exposure_adjustment is not None:
        exposure = gen_uniform([batch_size, 1], *exposure_adjustment)
        params['exposure'] = exposure

    if gamma_adjustment is not None:
        gamma = gen_uniform([batch_size, 1], *gamma_adjustment)
        params['gamma'] = gamma

    if brightness_adjustment is not None:
        brightness = gen_uniform([batch_size, 1], *brightness_adjustment)
        params['brightness'] = brightness

    if saturation_adjustment is not None:
        saturation = gen_uniform([batch_size, 1], *saturation_adjustment)
        params['saturation'] = saturation

    if hue_adjustment is not None:
        hue = gen_uniform([batch_size, 1], *hue_adjustment)
        params['hue'] = hue
    
    if contrast_adjustment is not None:
        contrast = gen_uniform([batch_size, 1], *contrast_adjustment)
        params['contrast'] = contrast
    
    if blur_adjustment is not None:
        blur = gen_uniform([batch_size, 1], *blur_adjustment['blur'])
        params['blur'] = blur

        sigma = gen_uniform([batch_size, 1], *blur_adjustment['sigma'])
        params['sigma'] = sigma
        
    if background_noise is not None:
        noise = 0
        res = background_noise['resolution']
        if 'std_grey' in background_noise:
            noise = noise + torch.randn(batch_size, 1, res, res) * background_noise['std_grey']
        if 'std_rgb' in background_noise:
            noise = noise + torch.randn(batch_size, 3, res, res) * background_noise['std_rgb']
        if 'std_grey' in background_noise or 'std_rgb' in background_noise:
            params['bg_noise'] = noise
            
    if background_color is not None:
        bg_color = 0
        if 'grey' in background_color:
            bg_color = bg_color + gen_uniform([batch_size, 1], *background_color['grey'])
        if 'rgb' in background_color:
            bg_color = bg_color + gen_uniform([batch_size, 3], *background_color['rgb'])
        if 'grey' in background_color or 'rgb' in background_color:
            params['bg_color'] = bg_color
            
    return params


def update_transform_parameters(params, update_params):

    if 'scale' in update_params: 
        scale = update_params['scale']
        params['scale'] = params.get('scale',1) * scale
    
    if 'translate' in update_params: 
        translate = update_params['translate']
        params['translate'] = params.get('translate',0) + translate

    if 'elastic' in update_params: 
        elastic = update_params['elastic']
        params['elastic'] = params.get('elastic',0) + elastic

    if 'exposure' in update_params: 
        exposure = update_params['exposure']
        params['exposure'] = params.get('exposure',1) * exposure

    if 'gamma' in update_params: 
        gamma = update_params['gamma']
        params['gamma'] = params.get('gamma',1) * gamma

    if 'saturation' in update_params: 
        saturation = update_params['saturation']
        params['saturation'] = params.get('saturation',1) * saturation

    if 'hue' in update_params: 
        hue = update_params['hue']
        params['hue'] = params.get('hue',0) + hue
    
    if 'contrast' in update_params: 
        contrast = update_params['contrast']
        params['contrast'] = params.get('contrast',1) * contrast
    
    if 'blur' in update_params or 'sigma' in update_params: 
        blur = update_params['blur']
        params['blur'] = params.get('blur',0) * (1 - blur) + blur

        sigma = update_params['sigma']
        params['sigma'] = (params.get('sigma',0)**2 + sigma**2)**0.5     
        
    if 'bg_noise' in update_params:
        params['bg_noise'] = params.get('bg_noise',0) + update_params['bg_noise']
        
    if 'bg_color' in update_params:
        params['bg_color'] = params.get('bg_color',0) + update_params['bg_color']
        


class MVLRMAugmentor:

    def __init__(self, FLAGS):
        self.flags = FLAGS.get('image_augmentation', {})

    def __call__(self, rgb, albedo, normal, xyz, mask):
        '''
        performs in-place data augmentation on rendered images and intrinsic matirces

        rgb: rgb images of shape (N,3,H,W) noramlized in range [0,1]
        albedo: albedo images of shape (N,3,H,W) noramlized in range [0,1]
        normal: world normal images of shape (N,3,H,W) noramlized in range [0,1]
        xyz: world xyz coordinates images of shape (N,3,H,W) noramlized in range [0,1]
        mask: alpha images of shape (N,1,H,W) in range [0,1]
        intrinsic: camera intrinsic matrices of shape (N,3,3)
        '''

        flags = self.flags
        n_views = rgb.shape[0]

        augmentation_params = dict(rgb={}, albedo={}, normal={}, xyz={}, mask={})

        # generate random transform parameters
        for augmentation in flags:

            if augmentation.get('multiview', False):
                batch_size = 1
            else:
                batch_size = n_views

            param_updates = gen_transform_parameters(batch_size, **augmentation)

            for modality in augmentation.get('modalities', ['rgb', 'albedo', 'normal', 'xyz', 'mask']):
                update_transform_parameters(augmentation_params[modality], param_updates)

        # apply random transform
        rgb[:], rgb_transformed = transform(rgb, **augmentation_params['rgb'])
        rgb[:], rgb_adjusted = adjust(rgb, **augmentation_params['rgb'])

        albedo[:], albedo_transformed = transform(albedo, **augmentation_params['albedo'])
        albedo[:], albedo_adjusted = adjust(albedo, **augmentation_params['albedo'])

        normal[:], normal_transformed = transform(normal, **augmentation_params['normal'])
        normal[:], normal_adjusted = adjust(normal[:], **augmentation_params['normal'])

        xyz[:], xyz_transformed = transform(xyz, **augmentation_params['xyz'])
        xyz[:], xyz_adjusted = adjust(xyz[:], **augmentation_params['xyz'])

        mask[:], mask_transfotmed = transform(mask, **augmentation_params['mask'])
        mask, mask_adjusted = transform(torch.cat([mask]*3, dim=1), **augmentation_params['mask'])
        mask = mask[:,0:1]

        augmentation_params['rgb']['transformed'] = rgb_transformed
        augmentation_params['rgb']['adjusted'] = rgb_adjusted
        augmentation_params['albedo']['transformed'] = albedo_transformed
        augmentation_params['albedo']['adjusted'] = albedo_adjusted
        augmentation_params['normal']['transformed'] = normal_transformed
        augmentation_params['normal']['adjusted'] = normal_adjusted
        augmentation_params['xyz']['transformed'] = xyz_transformed
        augmentation_params['xyz']['adjusted'] = xyz_adjusted
        augmentation_params['mask']['transformed'] = mask_transfotmed
        augmentation_params['mask']['adjusted'] = mask_adjusted
        
        # clip values to [0,1]

        rgb.clamp_(0,1)
        albedo.clamp_(0,1)
        normal.clamp_(0,1)
        xyz.clamp_(0,1)
        mask.clamp_(0,1)
        
        return augmentation_params
        

        




