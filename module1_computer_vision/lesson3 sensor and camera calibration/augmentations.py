import copy
import random
import numpy as np 
from PIL import Image


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    - [xmin, ymin, xmax, ymax]
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union, [xmin, ymin, xmax, ymax]


def hflip(img, bboxes):
    """
    horizontal flip of an image and annotations
    args:
    - img [PIL.Image]: original image
    - bboxes [list[list]]: list of bounding boxes
    return:
    - flipped_img [PIL.Image]: horizontally flipped image
    - flipped_bboxes [list[list]]: horizontally flipped bboxes
    """
    w, h = img.size()
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_bboxes = [[w-bbox[2],bbox[1],w-bbox[0],bbox[3]] for bbox in bboxes]
    return flipped_img, flipped_bboxes


def resize(img, boxes, size):
    """
    resized image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - size [array]: 1x2 array [width, height]
    returns:
    - resized_img [PIL.Image]: resized image
    - resized_boxes [list[list]]: resized bboxes
    """
    W, H = img.size()
    w, h = size
    wf, hf = w / W, h / H
    resized_image = img.resize(size)
    resized_boxes = [[wf*box[0],hf*box[1],wf*box[2],hf*box[3]] for box in boxes]
    return resized_image, resized_boxes


def random_crop(img, boxes, classes, crop_size, min_area=100):
    """
    random cropping of an image and annotations
    args:
    - img [PIL.Image]: original image
    - boxes [list[list]]: list of bounding boxes
    - classes [list[int]]: list of class ids
    - crop_size [array]: 1x2 array [width, height]
    - min_area [int]: min area of a bbox to be kept in the crop
    returns:
    - cropped_img [PIL.Image]: resized image
    - cropped_boxes [list[list]]: resized bboxes
    - cropped_classes [list[int]]: resized classes
    """
    W, H = img.size()
    w, h = crop_size
    assert w <= W and h <= H
    randx, randy = random.randint(1, W-w), random.randint(1, H-h)
    crop_box = [randx, randy, randx+w, randy+h]  # left, top, right, bottom
    cropped_img = img.crop(crop_box)
    cropped_boxes = []
    cropped_classes = []
    for box, clazz in zip(boxes, classes):
        iou, [xmin, ymin, xmax, ymax] = calculate_iou(crop_box, box)
        if (xmax-xmin) * (ymax-ymin) < min_area:
            continue
        cropped_boxes.append([xmin-randx,ymin-randy,xmax-randx,ymax-randy])
        cropped_classes.append(clazz)
    return cropped_image, cropped_boxes, cropped_classes


if __name__ == '__main__":
    # fix seed to check results
    
    # open annotations
    
    # filter annotations and open image
    
    # check horizontal flip, resize and random crop
    # use check_results defined in utils.py for this
    pass