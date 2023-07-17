'''
Author: lorenzo · jhsun@storagex.ai
Date: 2023-05-16 18:08:56
LastEditors: lorenzo · jhsun@storagex.ai
LastEditTime: 2023-05-16 20:24:04
Description: 

Copyright (c) 2023 by StorageX, All Rights Reserved. 
'''
import torch
from collections import OrderedDict
import os.path as osp
import numpy as np
from typing import Dict, List, Optional
import cv2
from prettytable import PrettyTable
from glob import glob


def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                        num_classes: int, ignore_index: int):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        label (torch.tensor): Ground truth segmentation map
            or label filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_area_to_metrics(total_area_intersect: np.ndarray,
                          total_area_union: np.ndarray,
                          total_area_pred_label: np.ndarray,
                          total_area_label: np.ndarray,
                          metrics: List[str] = ['mIoU'],
                          nan_to_num: Optional[int] = None,
                          beta: int = 1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (np.ndarray): The intersection of prediction
            and ground truth histogram on all classes.
        total_area_union (np.ndarray): The union of prediction and ground
            truth histogram on all classes.
        total_area_pred_label (np.ndarray): The prediction histogram on
            all classes.
        total_area_label (np.ndarray): The ground truth histogram on
            all classes.
        metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
            'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be
            replaced by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
    Returns:
        Dict[str, np.ndarray]: per category evaluation metrics,
            shape (num_classes, ).
    """

    def f_score(precision, recall, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | torch.Tensor): The precision value.
            recall (float | torch.Tensor): The recall value.
            beta (int): Determines the weight of recall in the combined
                score. Default: 1.

        Returns:
            [torch.tensor]: The f-score value.
        """
        score = (1 + beta**2) * (precision * recall) / (
            (beta**2 * precision) + recall)
        return score

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metrics {metrics} is not supported')

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor([
                f_score(x[0], x[1], beta) for x in zip(precision, recall)
            ])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


def compute_metrics(results: list, class_names: list) -> Dict[str, float]:
    results = tuple(zip(*results))
    assert len(results) == 4
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(
        total_area_intersect, total_area_union, total_area_pred_label,
        total_area_label, ['mIoU'], None, 1)
    
    ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
    metrics = dict()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            metrics[key] = val
        else:
            metrics['m' + key] = val

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)
    print('\n' + class_table_data.get_string())
    return metrics


def calculate_miou_macc(ground_truth_folder, predicted_folder, num_classes, class_names):
    gt_dirs = glob(osp.join(ground_truth_folder, '*.png'))
    results = []

    for gt_dir in gt_dirs:  # Assuming the same number of samples in both folders
        img_name = osp.basename(gt_dir)[:-4]
        pred_dir = osp.join(predicted_folder, img_name + '.png')
        gt_img = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_dir, cv2.IMREAD_GRAYSCALE)
        pred_label = torch.from_numpy(pred_img)
        label = torch.from_numpy(gt_img)

        results.append(intersect_and_union(pred_label, label, num_classes, 255))

    metrics = [compute_metrics(results, class_names)]
    print(metrics)

if __name__ == "__main__":
    ground_truth_folder = '/data/dataset/data/weice_data/seg/20230707_AOI/val/masks/'
    predicted_folder = '/workspace/aoi_seg_v0.5/'
    num_classes = 3
    class_names = ('background', 'pad', 'nmpad')
    calculate_miou_macc(ground_truth_folder, predicted_folder, num_classes, class_names)