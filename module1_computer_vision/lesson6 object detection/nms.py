import json

from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    filtered = []
    iou_thres = 0.5
    # sort by scores
    # predictions = [[box, score, clazz] for box,score,clazz in zip(predictions['boxes'], predictions['scores'], predictions['classes'])]
    predictions = [[box, score] for box,score in zip(predictions['boxes'], predictions['scores'])]
    predictions = sorted(predictions, key=lambda a: a[1], reverse=True)
    # nms
    while predictions:
        pmax = predictions.pop(0)
        filtered.append(pmax)
        newpreds = []
        for i,p in enumerate(predictions):
            iou = calculate_iou(pmax[0], p[0])
            if iou < iou_thres:
                newpreds.append(p)
        predictions = newpreds
    return filtered


if __name__ == '__main__':
    with open('data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
    check_results(filtered)