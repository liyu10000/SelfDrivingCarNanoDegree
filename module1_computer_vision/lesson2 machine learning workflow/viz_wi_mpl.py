from utils import get_data

import os
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    n = len(ground_truth)
    nrows, ncols = 4, n//4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    image_dir = "./data/images"
    color_map = {1:'r', 2:'g'}
    for i, observation in enumerate(ground_truth):
        filename = observation["filename"]
        boxes = observation["boxes"]
        classes = observation["classes"]
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        r, c = i // ncols, i % ncols
        axes[r][c].axis('off')
        axes[r][c].imshow(image)
        for box, clazz in zip(boxes, classes):
            # x,y seem to have been flipped in inputs
            rect = mpl.patches.Rectangle((box[1], box[0]), (box[3]-box[1]), (box[2]-box[0]), linewidth=1, edgecolor=color_map[clazz], facecolor="none")
            axes[r][c].add_patch(rect)
    plt.savefig("./output.png", pad_inches=0, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)