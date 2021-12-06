import copy
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_iou, check_results


def calc_mAP(preds, gts):
    print(preds)
    print(gts)
    # transform both preds and gts and sort preds
    preds, gts = preds[0], gts[0]
    preds = [[box, clazz, score] for box,clazz,score in zip(preds['boxes'], preds['classes'], preds['scores'])]
    gts = [[box, clazz] for box,clazz in zip(gts['boxes'], gts['classes'])]
    preds = sorted(preds, key=lambda a: a[2], reverse=True)
    
    # get precision and recall per prediction
    TP = 0
    curve = []
    matched = []  # record matched gt boxes to avoid counting twice
    iou_thres = 0.5
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            if j in matched:
                continue
            if calculate_iou(pred[0], gt[0]) >= iou_thres and pred[1] == gt[1]:
                TP += 1
                matched.append(j)
        precision = TP / (i+1)
        recall = TP / len(gts)
        curve.append([precision, recall])
        
    # smooth Precision-Recall (PR) curve (VOC2010-2012 metric)
    curve = np.array(curve)
    ct = Counter(curve[:, 1])
    boundaries = sorted([k for k,v in ct.items() if v > 1])
    # get max precision values
    maxes = []
    for i in range(len(boundaries)):
        if i != len(boundaries) - 1:
            loc = [p[0] for p in curve if boundaries[i+1] >= p[1] > boundaries[i]]
            maxes.append(np.max(loc))
        else:
            loc = [p[0] for p in curve if p[1] > boundaries[i]]
            maxes.append(np.max(loc))
    smoothed = copy.copy(curve)
    replace = -1
    for i in range(smoothed.shape[0]-1):
        if replace != -1:
            smoothed[i, 0] = maxes[replace]
        if smoothed[i, 1] == smoothed[i+1, 1]:
            replace += 1 
    print(curve)
    print(smoothed)

    # calculate mAP
    cmin = 0
    mAP = 0
    for i in range(smoothed.shape[0] - 1):
        if smoothed[i, 1] == smoothed[i+1, 1]:
            mAP += (smoothed[i, 1] - cmin) * smoothed[i, 0]
            cmin = smoothed[i, 1]
    mAP += (smoothed[-1, 1] - cmin) * smoothed[-1, 0]

    # plot original and smoothed PR curves
    plt.plot(curve[:, 1], curve[:, 0], linewidth=4)
    plt.plot(smoothed[:, 1], smoothed[:, 0], linewidth=4)
    plt.xlabel('recall', fontsize=18)
    plt.ylabel('precision', fontsize=18)
    plt.show()
    
    return mAP


if __name__ == '__main__':
    # load data 
    with open('data/predictions.json', 'r') as f:
        preds = json.load(f)

    with open('data/ground_truths.json', 'r') as f:
        gts = json.load(f)
    
    mAP = calc_mAP(preds, gts)
    check_results(mAP)