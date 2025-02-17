import torch
import torchvision.transforms.functional as F
import random

def to_2_tuple(input):
    """
    Accepts an integer or a 2-tuple and returns a 2-tuple.

    Inputs:
    - input (int or tuple): An integer or a 2-tuple.

    Returns:
    - tuple: A 2-tuple based on the input.
    """
    if isinstance(input, int):
        return (input, input)
    elif isinstance(input, tuple) and len(input) == 2:
        return input
    else:
        raise ValueError("Input must be an integer or a 2-tuple.")

def random_crop_image(image, intrinsic, crop_size):
    """
    Randomly crops a single image and adjust its intrinsic matrix.

    Inputs:
    - image (torch.Tensor): An images with shape (C, H, W).
    - intrinsic (torch.Tensor): intrinsic matrix with shape (3, 3).
    - crop_size (int or tuple): int h=w or tuple of (h,w), size of cropped image

    Returns:
    - cropped_image (torch.Tensor): Cropped image with shape (C, h, w).
    - new_intrinsic (torch.Tensor): Adjusted intrinsic matrix with shape (3, 3).
    """

    # Get original height and width
    C, H, W = image.shape
    h, w = to_2_tuple(crop_size)

    assert image.ndim == 3 and intrinsic.ndim == 2
    assert H >= h and W >= w, "crop_size cannot exceed original image size"

    # Initialize tensor to hold the new intrinsic matrices
    new_intrinsic = intrinsic / intrinsic[2,2]

    # Calculate top and left positions for the crop
    top = random.randint(0, H-h)
    left = random.randint(0, W-w)

    # Crop images
    cropped_image = image[:, top:top+h, left:left+w]

    # Adjust the intrinsic matrices
    # For the intrinsic matrix, we need to adjust the optical center coordinates
    new_intrinsic[0, 2] -= left
    new_intrinsic[1, 2] -= top

    return cropped_image, new_intrinsic

def resize(image, intrinsic, new_size, align_corners=False, mode='nearest', antialias=False):
    """
    Resize a single image and adjust its intrinsic matrix.

    Inputs:
    - image (torch.Tensor): An images with shape (C, H, W).
    - intrinsic (torch.Tensor): intrinsic matrix with shape (3, 3).
    - new_size (int or tuple): int h=w or tuple of (h,w), size of new image
    - align_corners (boolean): whether to align corners. see torch.nn.functional.interpolate
    - mode (str): can be 'nearest', 'bilinear', 'bicubic' or 'area'. see torch.nn.functional.interpolate
    - antialias (boolean): whether to antialias, this is only supported for 'bilinear' and 'bicubic' modes. see torch.nn.functional.interpolate

    Returns:
    - resized_image (torch.Tensor): Resized image with shape (C, h, w).
    - new_intrinsic (torch.Tensor): Adjusted intrinsic matrix with shape (3, 3).
    """

    # Get original height and width
    C, H, W = image.shape
    h, w = to_2_tuple(new_size)

    assert image.ndim == 3 and intrinsic.ndim == 2
    if align_corners:
        assert H > 1 and W > 1 and h > 1 and w > 1, "image sizes must be greater than 1 if corners are aligned"

    # Initialize tensor to hold the new intrinsic matrices
    new_intrinsic = intrinsic / intrinsic[2,2]
    
    if align_corners:
        mode = 'bilinear'
    else:
        align_corners = None

    # Crop images
    resized_image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(h,w), mode=mode, align_corners=align_corners, antialias=antialias).squeeze(0)

    # Adjust the intrinsic matrices
    if align_corners:
        height_scale, width_scale = (h-1) / (H-1), (w-1) / (W-1)
    else:
        height_scale, width_scale = h / H, w / W

    # Update intrinsic matrices
    new_intrinsic[0, 0] *= width_scale  # Scale x focal length
    new_intrinsic[1, 1] *= height_scale # Scale y focal length
    new_intrinsic[0, 2] = (new_intrinsic[0, 2] + 0.5) * width_scale - 0.5
    new_intrinsic[1, 2] = (new_intrinsic[1, 2] + 0.5) * height_scale - 0.5

    return resized_image.unsqueeze(dim=0), new_intrinsic.unsqueeze(dim=0)


if __name__ == "__main__":

    img = torch.randn((3, 512, 512))
    intrinsic = torch.tensor([
        [400, 0, 255.5],
        [0, 400, 255.5],
        [0,   0,   1.0]
    ])

    img, intrinsic = resize(img, intrinsic, new_size=384, align_corners=True, mode='bilinear')
    img, intrinsic = random_crop_image(img, intrinsic, crop_size=128)

    '''
    TODO list for yunfei:
        - add resize and crop in dataloader pipeline
        - allow optional resize and/or crop of NON-REFERENCE images, if both then first resize then crop
        - allow optional resize of reference images
        - expose whether to resize and/or crop and their parameters in config file
    '''
