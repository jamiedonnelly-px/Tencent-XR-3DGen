import imgaug.augmenters as iaa
import cv2

imgaug_seq = iaa.Sequential([
    iaa.PiecewiseAffine(scale=(0.04, 0.04), nb_rows=(2, 4), nb_cols=(2, 4))
])

image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/selected_highres_may/image-75.png"
image = cv2.imread(image_path)
image_aug = imgaug_seq(images=image[None, ...])
cv2.imwrite("img_aug.png", image_aug[0])