import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(
        description='LTCV compare predicted results with groundtruth results')
    parser.add_argument(
        '--groundtruth',
        type=str,
        default='/workspace/weice_data/new_ann.json',
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--predictions',
        type=str,
        default='/workspace/weice_data/output.json',
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    args = parser.parse_args()
    return args


def main():
    args = parse_args() 
    ground_truth_file = args.groundtruth
    predictions_file = args.predictions

    # initialize COCO ground truth api
    cocoGt = COCO(ground_truth_file)

    # initialize COCO detections api
    cocoDt = cocoGt.loadRes(predictions_file)

    imgIds=sorted(cocoGt.getImgIds())
    annType = 'bbox'

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    # optional - category1 only, see cocoGt.loadCats(cocoGt.getCatIds())
    #cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # print(cocoEval.stats)

if __name__ == '__main__':
    main()
