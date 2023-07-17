from pycocotools.coco import COCO
import json
# import pickle


# Compute the intersection over union (IoU) of two bounding boxes
def iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin + 1)
    h = max(0, ymax - ymin + 1)
    intersection = w * h
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    union = area1 + area2 - intersection
    return intersection / union

# Compute the recall and precision for the given class
def evaluate(gt_boxes, gt_labels, pred_boxes, pred_labels, cls_id, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        if gt_label != cls_id:
            continue
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            if pred_label != cls_id:
                continue
            if iou(gt_box, pred_box) >= iou_threshold:
                true_positives += 1
                break
        else:
            false_negatives += 1
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        if pred_label != cls_id:
            continue
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            if gt_label != cls_id:
                continue
            if iou(gt_box, pred_box) >= iou_threshold:
                break
        else:
            false_positives += 1
    # recall = true_positives / (true_positives + false_negatives)
    # precision = true_positives / (true_positives + false_positives)
    return true_positives, false_positives, false_negatives


def eval(coco_data,prediction_data, label, score_threshold=0.1, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    # Compute the recall and precision for the given class
    gt_boxes = coco_data["bbox"]  # List of ground truth bounding boxes
    gt_labels = coco_data["image_id"] # List of ground truth labels
    pred_boxes_orin = prediction_data["bbox"] # List of predicted bounding boxes
    pred_labels_orin = prediction_data["image_id"] # List of predicted labels
    pred_scores = prediction_data["score"]
    cls_id = label  # ID of the class to evaluate

    pred_boxes = list()
    pred_labels = list()
    assert len(pred_boxes_orin) == len(pred_labels_orin)
    for i in range(len(pred_boxes_orin)):
        if pred_scores[i] >= score_threshold:
            pred_boxes.append(pred_boxes_orin[i])
            pred_labels.append(pred_labels_orin[i])

    true_positives, false_positives, false_negatives = evaluate(gt_boxes, gt_labels, pred_boxes, pred_labels, cls_id, iou_threshold=0.5)

    return true_positives, false_positives, false_negatives



# 加载预测结果和真实标签
prediction_files = "/workspace/output.json"
annotations_file = "/data/dataset/data/weice_data/det/val_labels.json"

with open(annotations_file, "r") as file:
    coco_datas = json.load(file)

# Open the Pickle file
with open(prediction_files, "r") as file:
    prediction_datas  = json.load(file)

score_thresholds = [0.001, 0.0097, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
recalls = list()
precisions = list()
for score_threshold in score_thresholds:
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    for coco_data in coco_datas["annotations"]:
        for prediction_data in prediction_datas:
            for label in [1, 2, 3, 4]:
                true_positives, false_positives, false_negatives = eval(coco_data,prediction_data, label, score_threshold=score_threshold, iou_threshold=0.1)
                all_true_positives += true_positives
                all_false_positives += false_positives
                all_false_negatives += false_negatives

    recall = all_true_positives / (all_true_positives + all_false_negatives) if (all_true_positives + all_false_negatives) > 0 else 0.0
    precision = all_true_positives / (all_true_positives + all_false_positives) if (all_true_positives + all_false_positives) > 0 else 0.0
    recalls.append(recall)
    precisions.append(precision)

# 打印结果
print(recalls)
print(precisions)

import matplotlib.pyplot as plt

# 在这里定义您的 precision 和 recall 数据
# precision = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# recall = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# 绘制 PR 曲线
plt.plot(recalls, precisions)

# 在图上标记数据点
for i in range(len(recalls)):
    plt.plot(recalls[i], precisions[i], "ro")
    plt.annotate("(%.4f, %.4f, %.4f)" % (score_thresholds[i], recalls[i], precisions[i]), xy=(recalls[i], precisions[i]))

# 添加图标标题和坐标轴标题
plt.title("PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

# 显示图表
plt.show()

# import matplotlib.pyplot as plt
#
# # 定义 x 和 y 轴的数据
# x = recalls
# y = precisions
#
# # 绘制折线图
# plt.plot(x, y)
#
# # 在图上标记数据点
# for i in range(len(x)):
#     plt.plot(x[i], y[i], "ro")
# #     plt.annotate("(%.4f, %.4f, %.4f)" % (score_thresholds[i], x[i], y[i]), xy=(x[i], y[i]))
#
# # 添加图标标题和坐标轴标题
# plt.title("My plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
#
# # 显示图表
# plt.show()