import torch
import torch.nn.functional as F
import math
import numpy as np


def adjust_gamma(images, gamma, gain):
    """
    Adjust the hue, saturation, and brightness of a batch of images.
    
    Parameters:
    - images: Tensor of shape [batch, 3, H, W] with pixel values in range [0, 1].
    - gamma: Tensor of shape [batch or 1, 1] for gamma adjustment.
    - gain: Tensor of shape [batch or 1, 1] for exposure adjustment.

    Returns:
    - adjusted_images: Tensor of shape [batch, 3, H, W] with pixel values in range [0, 1].
    """
    # Convert RGB to HSV
    gamma = gamma.view(-1, 1, 1, 1)
    gain = gain.view(-1, 1, 1, 1)

    adjusted_images = gain * torch.pow(images, gamma)

    adjusted_images = torch.clamp(adjusted_images, 0, 1)  # Ensure values stay within [0, 1]

    return adjusted_images

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

def adjust_hsv(images, brightness_factor, saturation_factor, hue_factor):
    """
    Adjust the hue, saturation, and brightness of a batch of images.
    
    Parameters:
    - images: Tensor of shape [batch, 3, H, W] with pixel values in range [0, 1].
    - brightness_factor: Tensor of shape [batch or 1, 1] for brightness adjustment.
    - saturation_factor: Tensor of shape [batch or 1, 1] for saturation adjustment.
    - hue_factor: Tensor of shape [batch or 1, 1], values in range [-0.5, 0.5] for hue adjustment.

    Returns:
    - adjusted_images: Tensor of shape [batch, 3, H, W] with pixel values in range [0, 1].
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv(images)
    
    # Adjust hue
    hsv[:, 0, :, :] += hue_factor.unsqueeze(-1)
    hsv[:, 0, :, :] = torch.remainder(hsv[:, 0, :, :], 1)  # Ensure hue stays within [0, 1]

    # Adjust saturation
    hsv[:, 1, :, :] *= saturation_factor.unsqueeze(-1)
    hsv[:, 1, :, :] = torch.clamp(hsv[:, 1, :, :], 0, 1)  # Ensure saturation stays within [0, 1]

    # Convert HSV back to RGB
    adjusted_images = hsv_to_rgb(hsv)

    # Adjust brightness
    adjusted_images *= brightness_factor.unsqueeze(-1).unsqueeze(-1)
    adjusted_images = torch.clamp(adjusted_images, 0, 1)  # Ensure values stay within [0, 1]

    return adjusted_images


def adjust_contrast(images, contrast_factors):
    """
    Adjust the contrast of a batch of images.

    Parameters:
    - images: Tensor of shape [batch, channel, H, W] with pixel values in range [0, 1].
    - contrast_factors: Tensor of shape [batch, 1] or [1, 1] for contrast adjustment, 
                        with >1 to increase contrast, <1 to decrease contrast, and 1 for no change.

    Returns:
    - adjusted_images: Tensor of shape [batch, channel, H, W] with adjusted contrast.
    """
    # Ensure contrast_factors is broadcastable to the batch size
    contrast_factors = contrast_factors.view(-1, 1, 1, 1)

    # Calculate the mean pixel value for each image
    mean_vals = images.mean(dim=[1, 2, 3], keepdim=True)

    # Adjust the contrast
    # (image - mean) * contrast_factor + mean: Shift, scale, then shift back
    adjusted_images = (images - mean_vals) * contrast_factors + mean_vals

    # Clip the values to ensure they remain in the [0, 1] range
    adjusted_images = torch.clamp(adjusted_images, 0, 1)

    return adjusted_images


def generate_1d_gaussian_kernel(sigmas, kernel_size):
    """
    Generate a batch of 1D Gaussian kernels.
    
    Parameters:
    - sigmas: Tensor of shape [batch, 1], the sigma values for the Gaussian kernel.
    - kernel_size: Integer, the size of the kernel.
    
    Returns:
    - A batch of 1D Gaussian kernels, shape: [batch, 1, kernel_size]
    """
    x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    x_grid = x.repeat(sigmas.size(0), 1)  # Shape: [batch, kernel_size]
    
    kernels = torch.exp(-0.5 * (x_grid ** 2) / sigmas ** 2) # [batch, kernel_size]
    kernels /= kernels.sum(dim=-1, keepdim=True)  # Normalize

    return kernels.unsqueeze(1) # [batch, 1, kernel_size]

def apply_separate_convolution(images, kernels):
    """
    Apply batched convolution using the generated 1d kernels once in width and once in height dimension.
    
    Parameters:
    - images: Tensor of shape [batch, channels, height, width]
    - kernels: Tensor of shape [batch, 1, kernel_size] for 1D Gaussian kernels
    
    Returns:
    - Blurred images: Tensor of shape [batch, channels, height, width]
    """
    batch, channels, h, w = images.shape
    padding = kernels.shape[-1] // 2

    # Adjust kernels to match the channel dimension of the images
    kernel_x = kernels.expand(batch, channels, -1).reshape(-1, 1, kernels.size(-1), 1)  # Shape: [batch * channels, 1, kernel_size, 1]
    kernel_y = kernels.expand(batch, channels, -1).reshape(-1, 1, 1, kernels.size(-1))  # Shape: [batch * channels, 1, 1, kernel_size]

    # Apply convolution in x direction

    padded_images = F.pad(images, (padding, padding, padding, padding), mode='replicate') # border padding

    blurred_x = F.conv2d(padded_images.reshape(1, -1, padded_images.size(2), padded_images.size(3)), 
                         kernel_x, padding='valid', groups=batch*channels) # shape [1, batch * channels, h, w]
    
    # Apply convolution in y direction
    blurred_images = F.conv2d(blurred_x, kernel_y, padding='valid', groups=batch*channels)
    
    
    return blurred_images.view(batch, channels, images.size(2), images.size(3))

def gauss_blur(images, sigma):
    """
    Apply Gaussian blur on a batch of images using separable 1D convolutions,
    with sigma values relative to the image size.

    Parameters:
    - images: Tensor of shape [batch, c, h, w].
    - sigma: Tensor of shape [batch/1, 1], sigma values relative to image size.

    Returns:
    - Blurred images: Tensor of shape [batch, c, h, w].
    """
    _, _, h, w = images.shape

    # Convert relative sigma to absolute sigma by multiplying by the image dimensions
    sigma = sigma * max(h,w)

    # Determine maximum absolute sigma to define kernel size
    max_sigma = sigma.max().item()

    # Determine kernel size using the 3-sigma rule, ensuring it's odd
    kernel_size = int(math.ceil(max_sigma * 3) * 2 + 1)
    kernel_size = min(kernel_size, min(h, w))  # Ensure kernel size does not exceed image dimensions

    # Create separable Gaussian kernel
    kernel1d = generate_1d_gaussian_kernel(sigma, kernel_size).to(images.device)

    # Apply separable Gaussian blur
    blurred_images = apply_separate_convolution(images, kernel1d)

    return blurred_images

def adjust_gauss_blur(images, blur_factor, sigma): 
    """
    Adjust the blur or sharpness of a batch of images.

    Parameters:
    - images: Tensor of shape [batch, channel, H, W] with pixel values in range [0, 1].
    - blur_factor: Tensor of shape [batch] or [1] for blur adjustment,
                        with 0 for original image and 1 for total blurring. Use negative values for image shapening.
    - sigma: Tensor of shape [batch/1, 1], standard deviation of the Gaussian kernel for blurring relative to image size.

    Returns:
    - Tensor of shape [batch, channel, H, W] with adjusted sharpness.
    """
    # Ensure blur_factor is broadcastable to the batch size
    blur_factor = blur_factor.view(-1, 1, 1, 1)

    # Apply Gaussian blur
    blurred_images = gauss_blur(images, sigma)

    # Adjust the sharpness
    adjusted_images = images * (1-blur_factor) + blurred_images * blur_factor

    # Clip the values to ensure they remain in the [0, 1] range
    adjusted_images = torch.clamp(adjusted_images, 0, 1)

    return adjusted_images




