import cv2
import PIL
import random
import numpy as np

from PIL import Image

def fixed_thresh(value, threshold):
    '''
    固定阈值对单个像素点的二值化
    :param value: 像素灰度值
    :param threshold: 固定阈值
    :return value: 二值化后的像素灰度值
    '''
    if value >= threshold:
        value = 255
    else:
        value = 0
    return value


def sobel(image, threshold=100):
    '''
    利用Sobel算子的边缘检测
    :param img: 灰度图数组
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''

    if image.shape[-1] == 4:
        img = image[:, :, :3]
    else:
        img = image

    # Converting it into grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (img - img.min()) * 255.0 / (img.max() - img.min() + 1e-6)
    img = np.pad(img, (2, 2), mode='constant', constant_values=0)
    row, col = img.shape
    resultimg = np.zeros([row, col], np.float32)

    # 定义Sobel算子
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    for i in range(row - 2):
        for j in range(col - 2):
            grad_x = abs(np.sum(sobel_x * img[i:i + 3, j:j + 3]))
            grad_y = abs(np.sum(sobel_y * img[i:i + 3, j:j + 3]))
            resultimg[i + 1, j + 1] = 255 - (grad_x ** 2 + grad_y ** 2) ** 0.5
            # 二值化
            if threshold > 0:
                resultimg[i + 1, j + 1] = fixed_thresh(resultimg[i + 1, j + 1], threshold)
    resultimg = resultimg[2:row - 2, 2:col - 2]

    if image.shape[-1] == 4:
        resultimg = np.concatenate([resultimg[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        resultimg = np.concatenate([resultimg[..., None]]*3, axis=-1)
    return resultimg


def NMS(grad, dx, dy):
    '''
    Canny第四步，非极大值抑制
    '''
    row, col = grad.shape
    nms = np.zeros([row, col])

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if grad[i, j] != 0:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = grad[i, j]
                if np.abs(gradY) > np.abs(gradX):
                    w = np.abs(gradX) / (np.abs(gradY) + 1e-6)
                    grad2 = grad[i - 1, j]
                    grad4 = grad[i + 1, j]
                    if gradX * gradY > 0:
                        grad1 = grad[i - 1, j - 1]
                        grad3 = grad[i + 1, j + 1]
                    else:
                        grad1 = grad[i - 1, j + 1]
                        grad3 = grad[i + 1, j - 1]
                else:
                    w = np.abs(gradY) / (np.abs(gradX) + 1e-6)
                    grad2 = grad[i, j - 1]
                    grad4 = grad[i, j + 1]
                    if gradX * gradY > 0:
                        grad1 = grad[i + 1, j - 1]
                        grad3 = grad[i - 1, j + 1]
                    else:
                        grad1 = grad[i - 1, j - 1]
                        grad3 = grad[i + 1, j + 1]

                gradTemp1 = w * grad1 + (1 - w) * grad2
                gradTemp2 = w * grad3 + (1 - w) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    nms[i, j] = gradTemp
    return nms


def double_thresh(nms, ratiomin, ratiomax):
    '''
    Canny第五步，双阈值选取，弱边缘判断
    '''
    row, col = nms.shape
    dt = np.zeros([row, col])
    threshmin = np.max(nms) * ratiomin
    threshmax = np.max(nms) * ratiomax
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if (nms[i, j] < threshmin):
                dt[i, j] = 0
            elif (nms[i, j] > threshmax):
                dt[i, j] = 1
            elif ((nms[i - 1, j - 1:j + 1] < threshmax).any() or (nms[i + 1, j - 1:j + 1] < threshmax).any()
                  or (nms[i, [j - 1, j + 1]] < threshmax).any()):
                dt[i, j] = 1
    dt = dt[1:row - 1, 1:col - 1]
    return dt


def canny(image, sigma=1.3, kernel_size=(7, 7), ratiomin=0.08, ratiomax=0.5):
    '''
    利用Canny算子的边缘检测
    :param img: 灰度图数组
    :param sigma: 高斯核函数中参数sigma    
    :param kernel_size: 高斯核大小(height,width)
    :param ratiomin: 低阈值比例
    :param ratiomax: 高阈值比例
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    if image.shape[-1] == 4:
        img = image[:, :, :3]
    else:
        img = image
    # Converting it into grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1.转灰度图（输入已转）
    img = (img - img.min()) * 255.0 / (img.max() - img.min() + 1e-6)

    # 2.高斯滤波
    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape
    filterimg = np.zeros([row, col], np.float32)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            filterimg[i + cH, j + cW] = np.sum(img[i:i + H, j:j + W] * Gaussian_kernel(H, W, sigma))
    filterimg = filterimg[H - 1:row - H + 1, W - 1:col - W + 1]

    # 3.计算梯度值和方向（利用Sobel算子）
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filterimg = np.pad(filterimg, (2, 2), mode='constant', constant_values=0)
    row, col = filterimg.shape
    dx = np.zeros([row, col], np.float32)
    dy = np.zeros([row, col], np.float32)
    grad = np.zeros([row, col], np.float32)
    for i in range(row - 2):
        for j in range(col - 2):
            dx[i + 1, j + 1] = abs(np.sum(sobel_x * filterimg[i:i + 3, j:j + 3]))
            dy[i + 1, j + 1] = abs(np.sum(sobel_y * filterimg[i:i + 3, j:j + 3]))
            grad[i + 1, j + 1] = (dx[i + 1, j + 1] ** 2 + dy[i + 1, j + 1] ** 2) ** 0.5
    dx = dx[1:row - 1, 1:col - 1]
    dy = dy[1:row - 1, 1:col - 1]
    grad = grad[1:row - 1, 1:col - 1]

    # 4.非极大值抑制：取像素点梯度方向的局部梯度最大值（在梯度更大的方向利用插值）
    nms = NMS(grad, dx, dy)  # 填充了(1,1)的0

    # 5.双阈值的选取，弱边缘判断
    resultimg = double_thresh(nms, ratiomin, ratiomax)
    resultimg = 255 - resultimg
    resultimg = (resultimg - resultimg.min()) * 255.0 / (resultimg.max() - resultimg.min() + 1e-6)
    row, col = resultimg.shape
    # resultimg = resultimg[1:row - 1, 1:col - 1]

    if image.shape[-1] == 4:
        resultimg = np.concatenate([resultimg[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        resultimg = np.concatenate([resultimg[..., None]]*3, axis=-1)
    return resultimg


def laplacian(image, operator=1, threshold=225):
    '''
    利用拉普拉斯算子的边缘检测
    :param img: 灰度图数组
    :param operator: {1:四邻域拉普拉斯算子, 2:八邻域拉普拉斯算子}
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''

    if image.shape[-1] == 4:
        img = image[:, :, :3]
    else:
        img = image

    # Converting it into grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = (img - img.min()) * 255.0 / (img.max() - img.min() + 1e-6)
    img = np.pad(img, (2, 2), mode='constant', constant_values=0)
    row, col = img.shape
    resultimg = np.zeros([row, col], np.float32)

    # 定义拉普拉斯算子
    if operator == 1:
        laplacian_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif operator == 2:
        laplacian_operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    for i in range(row - 2):
        for j in range(col - 2):
            resultimg[i + 1, j + 1] = 255 - abs(np.sum(laplacian_operator * img[i:i + 3, j:j + 3]))
            # 二值化
            if threshold > 0:
                resultimg[i + 1, j + 1] = fixed_thresh(resultimg[i + 1, j + 1], threshold)
    # resultimg = resultimg[3:row - 3, 3:col - 3]
    resultimg = resultimg[2:row - 2, 2:col - 2]

    if image.shape[-1] == 4:
        resultimg = np.concatenate([resultimg[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        resultimg = np.concatenate([resultimg[..., None]]*3, axis=-1)
    return resultimg


def LaplaGauss_kernel(H, W, sigma):
    '''
    生成Laplacian of Gaussian核
    :param H: 高斯核高度
    :param W: 高斯核宽度
    :param sigma: 高斯核函数中参数sigma
    :return kernel: 高斯核
    '''
    cH = H // 2
    cW = W // 2
    kernel = np.zeros([H, W], np.float32)
    for x in range(-cH, H - cH):
        for y in range(-cW, W - cW):
            norm2 = x ** 2 + y ** 2
            sigma2 = sigma ** 2
            kernel[x + cH, y + cW] = (norm2 / sigma2 - 2) * np.exp(-norm2 / (2 * sigma2))
    kernel /= kernel.sum()
    return kernel


def LoG(image, sigma=1.3, kernel_size=(7, 7), threshold=100):
    '''
    利用Laplacian of Gaussian算子的边缘检测
    :param img: 灰度图数组
    :param sigma: 高斯核函数中参数sigma
    :param kernel_size: 高斯核大小(height,width)
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    if image.shape[-1] == 4:
        img = image[:, :, :3]
    else:
        img = image
    # Converting it into grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hi, wi = img.shape
    img = (img - img.min()) * 255.0 / (img.max() - img.min() + 1e-6)

    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape

    resultimg = np.zeros([row, col], np.float32)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            resultimg[i + cH, j + cW] = np.sum(img[i:i + H, j:j + W] * LaplaGauss_kernel(H, W, sigma))
            # 二值化
            if threshold > 0:
                resultimg[i + cH, j + cW] = fixed_thresh(resultimg[i + cH, j + cW], threshold)
    
    hr, wr = resultimg.shape
    resultimg = resultimg[hr//2-hi//2:hr//2-hi//2+hi, wr//2-wi//2:wr//2-wi//2+wi]

    if image.shape[-1] == 4:
        resultimg = np.concatenate([resultimg[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        resultimg = np.concatenate([resultimg[..., None]]*3, axis=-1)
    return resultimg


def Gaussian_kernel(H, W, sigma):
    '''
    生成高斯核
    :param H: 高斯核高度
    :param W: 高斯核宽度
    :param sigma: 高斯核函数中参数sigma
    :return kernel: 高斯核
    '''
    cH = H // 2
    cW = W // 2
    kernel = np.zeros([H, W], np.float32)
    for x in range(-cH, H - cH):
        for y in range(-cW, W - cW):
            kernel[x + cH, y + cW] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    kernel /= kernel.sum()
    return kernel


def DoG(img, k=1.6, sigma=1, kernel_size=(5, 5), threshold=-1):
    '''
    利用Difference-of-Gaussians算子的边缘检测
    :param img: 灰度图数组
    :param k: DoG中参数k
    :param sigma: 高斯核函数中参数sigma
    :param kernel_size: 高斯核大小(height,width)
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    if image.shape[-1] == 4:
        img = image[:, :, :3]
    else:
        img = image
    # Converting it into grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hi, wi = img.shape
    img = (img - img.min()) * 255.0 / (img.max() - img.min() + 1e-6)

    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape

    resultimg = np.zeros([row, col], np.float32)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            G = Gaussian_kernel(H, W, sigma)
            Gk = Gaussian_kernel(H, W, k * sigma)
            resultimg[i + cH, j + cW] = 255 - np.sum(img[i:i + H, j:j + W] * (G - Gk))
            # 二值化
            if threshold > 0:
                resultimg[i + cH, j + cW] = fixed_thresh(resultimg[i + cH, j + cW], threshold)
    
    hr, wr = resultimg.shape
    resultimg = resultimg[hr//2-hi//2:hr//2-hi//2+hi, wr//2-wi//2:wr//2-wi//2+wi]
    
    if image.shape[-1] == 4:
        resultimg = np.concatenate([resultimg[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        resultimg = np.concatenate([resultimg[..., None]]*3, axis=-1)
    return resultimg


def soft_thresh(value, threshold, phi):
    '''
    对单个像素点的软阈值化
    :param value: 像素灰度值
    :param threshold: 固定阈值
    :param phi: Larger or smaller ϕ control the sharpness of the black/white transitions in the image
    :return value: 二值化后的像素灰度值
    '''
    if value >= threshold:
        value = 1
    else:
        value = 1 + np.tanh(phi * (value - threshold))
    return value


def XDoG(img, p=45, k=1.6, sigma=1, kernel_size=(5, 5), threshold=100, phi=0.025):
    '''
    利用Extended Difference-of-Gaussians算子的边缘检测
    :param img: 灰度图数组
    :param p: 调整两个高斯滤波器的权重
    :param k: 高斯核函数标准差k*sigma
    :param sigma: 高斯核函数中参数sigma
    :param kernel_size: 高斯核大小(height,width)
    :param threshold: 灰度值变为255的阈值
    :param phi: Larger or smaller ϕ control the sharpness of the black/white transitions in the image
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    if image.shape[-1] == 4:
        img = image[:, :, :3]
    else:
        img = image
    # Converting it into grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hi, wi = img.shape
    img = (img - img.min()) * 255.0 / (img.max() - img.min())

    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape

    resultimg = np.zeros([row, col], np.float32)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            G = Gaussian_kernel(H, W, sigma)
            Gk = Gaussian_kernel(H, W, k * sigma)
            resultimg[i + cH, j + cW] = np.sum(img[i:i + H, j:j + W] * ((1 + p) * G - p * Gk))
            resultimg[i + cH, j + cW] = 255 * soft_thresh(resultimg[i + cH, j + cW], threshold, phi)

    hr, wr = resultimg.shape
    resultimg = resultimg[hr//2-hi//2:hr//2-hi//2+hi, wr//2-wi//2:wr//2-wi//2+wi]

    # resultimg = resultimg[H:row - H, W:col - W]
    if image.shape[-1] == 4:
        resultimg = np.concatenate([resultimg[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        resultimg = np.concatenate([resultimg[..., None]]*3, axis=-1)
    return resultimg

def vertical_grad(src, color_start=1, color_end=0):
    x, y, w, h = cv2.boundingRect(np.array(src)[:, :, -1])
    imgh, imgw, c = src.shape
    # 创建一幅与原图片一样大小的透明图片
    grad_back_img = np.zeros((imgh, imgw, 3))

    w_start = random.randint(0, int(x))
    w_end = random.randint(int(w_start), int(imgw/2))
    w_change_length = w_end - w_start

    grad_img = np.ndarray((imgh, w_change_length, 3))

    # opencv 默认采用 BGR 格式而非 RGB 格式
    grad = float(color_start - color_end) / w_change_length

    for i in range(w_change_length):
        grad_img[:, i] = np.array([[[color_start - i * grad]]])
    grad_back_img[:, w_start:w_end] = grad_img
    return grad_back_img

def add_shadow(img):
    """
    img: np.array [h, w, 4]
    """
    mask_left = vertical_grad(img, color_start=1.0, color_end=0)
    mask_right = vertical_grad(img[:, ::-1], color_start=1.0, color_end=0)[:, ::-1]
    mask = mask_left + mask_right
    # cv2.imwrite("mask_left.png", (mask_left*255).clip(0, 255).astype(np.uint8))
    # cv2.imwrite("mask_right.png", (mask_right*255).clip(0, 255).astype(np.uint8))
    # cv2.imwrite("mask.png", (mask*255).clip(0, 255).astype(np.uint8))
    # breakpoint()
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    img[:, :, :3] = img[:, :, :3] * (1 - mask) + np.array([[[b, g, r]]]) * mask
    
    return img.clip(0, 255).astype(np.uint8)


def add_shadow_mask(img):
    mask = img[:, :, 3]
    random_kernel_size = random.randint(4, 20) * 2 + 1
    mask = cv2.GaussianBlur(mask, (random_kernel_size, random_kernel_size), 0) / 255.0
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    img[:, :, :3] = img[:, :, :3] * np.stack([mask]*3, -1) + np.array([[[b, g, r]]]) * (1 - np.stack([mask]*3, -1))
    return img.clip(0, 255).astype(np.uint8)


def sketch_aug(image):
    """素描增强

    Args:
        image (np.ndarray): 图片矩阵

    Returns:
        np.ndarray: 图片的矩阵表达
    """
    if image.shape[-1] == 4:
        rgb = image[:, :, :3]
    else:
        rgb = image
    # Converting it into grayscale
    gray_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Inverting the image
    inverted_image = 255 - gray_image

    # The pencil sketch
    kernel_size = random.randint(1, 20) * 2 + 1
    blurred = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 0)
    inverted_blurred = 255 - blurred
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    if image.shape[-1] == 4:
        pencil_sketch = np.concatenate([pencil_sketch[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        pencil_sketch = np.concatenate([pencil_sketch[..., None]]*3, axis=-1)

    return pencil_sketch

def gray_aug(image):
    """灰度图片增强

    Args:
        image (np.ndarray): 图片矩阵

    Returns:
        np.ndarray: 图片的矩阵表达
    """
    if image.shape[-1] == 4:
        rgb = image[:, :, :3]
    else:
        rgb = image
    gray_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    if image.shape[-1] == 4:
        gray_image = np.concatenate([gray_image[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        gray_image = np.concatenate([gray_image[..., None]]*3, axis=-1)
    return gray_image

def invert_aug(image):
    """反色图片增强

    Args:
        image (np.ndarray): 图片矩阵

    Returns:
        np.ndarray: 图片的矩阵表达
    """
    if image.shape[-1] == 4:
        rgb = image[:, :, :3]
    else:
        rgb = image

    # Converting it into grayscale
    gray_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Inverting the image
    inverted_image = 255 - gray_image

    if image.shape[-1] == 4:
        inverted_image = np.concatenate([inverted_image[..., None]]*3+ [image[:, :, -1:]], axis=-1)
    else:
        inverted_image = np.concatenate([inverted_image[..., None]]*3, axis=-1)
    return inverted_image

def gray_scketch_aug(rgba_pil, prob=0.5):
    """灰度素描反相等增强的集成

    Args:
        rgba_pil (PIL.Image): 输入图片
        prob (float, optional): 增强概率. Defaults to 0.5.

    Returns:
        PIL.Image: 增强后图片
    """
    if random.random() > prob:
        return rgba_pil
    if isinstance(rgba_pil, PIL.Image.Image):
        if np.array(rgba_pil).shape[-1] == 4:
            bgra = cv2.cvtColor(np.array(rgba_pil).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
        elif np.array(rgba_pil).shape[-1] == 3:
            bgra = cv2.cvtColor(np.array(rgba_pil).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        bgra = rgba_pil

    rand_idx = random.choice([0,1,4])
    if rand_idx == 0:
        result = sketch_aug(bgra)
    elif rand_idx == 1:
        result = gray_aug(bgra)
    # elif rand_idx == 2:
    #     threshold = random.randint(120, 200)
    #     result = sobel(bgra, threshold=threshold)
    # elif rand_idx == 3:
    #     threshold = random.randint(225, 250)
    #     result = laplacian(bgra, operator=1, threshold=threshold)
    elif rand_idx == 4:
        result = invert_aug(bgra)

    if result.shape[-1] == 4:
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA).astype(np.uint8))
    elif result.shape[-1] == 3:
        result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.uint8))

    return result
