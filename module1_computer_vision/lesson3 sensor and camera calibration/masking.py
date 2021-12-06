from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    img = Image.open(path)
    img = np.asarray(img)
    mask = np.where(img > color_threshold, 1, 0)
    return img, mask


def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(img)
    axs[1].imshow(mask * 255)
    axs[2].imshow(img * mask)
    plt.savefig("./output.png", pad_inches=0, bbox_inches="tight")
    plt.close("all")



if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)