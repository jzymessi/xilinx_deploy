import pickle
import torch

# /local/lorenzo/Detection/datasets/weice/images/det_labeled/

pkl_result = []
with open('/workspace/aoi_det_v0.4.2.2_recall.txt', 'rb') as f:
    infer_results = f.readlines()
    for infer_result in infer_results:
        infer_result = infer_result.strip()
        result = eval(infer_result)
        img_path, img_id, bboxes, scores, labels = result["image_path"], result["image_id"], \
            result["pred_instances"]["boxes"], result["pred_instances"]["scores"], result["pred_instances"]["labels"]
        new_path = "/local/lorenzo/Detection/datasets/weice/images/" + img_path.split("det/")[-1]
        cur_dict = {
            'image_shape': (640,640),
            'img_path': new_path,
            'batch_input_shape': (640,640),
            'scale_factor': (0.625,0.625),
            'ori_shape': (768,1024),
            'img_id': img_id,
            'pad_shape': (640,640),
            'pred_instances':{
                'scores': torch.tensor(scores),
                'bboxes': torch.tensor(bboxes),
                'labels': torch.tensor(labels)
            }
            
        }
        pkl_result.append(cur_dict)

with open('/workspace/aoi_det_v0.4.2_2_recall.pkl', 'wb') as fp:
    pickle.dump(pkl_result, fp)

# with open('/workspace/output_recall_all.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

# print(loaded_data[0])