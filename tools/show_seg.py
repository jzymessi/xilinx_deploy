from PIL import Image
import glob
import numpy as np
import os
folder_path = '/workspace/aoi_seg_0705'

def get_image_files(directory):
    file_types = ['*.jpg', '*.jpeg', '*.png', '*.gif']  # 定义支持的图片文件类型
    image_files = []
    
    for file_type in file_types:
        image_files.extend(glob.glob(directory + '/' + file_type))
    
    return image_files

image_files = get_image_files(folder_path)

for file in image_files:
    img = Image.open(file)
    mask_np = np.array(img)
    mask_np[mask_np==1] = 100
    mask_np[mask_np==2] = 200
    
    file_name = os.path.basename(file)  # 提取文件名
    save_path = os.path.join("aoi_seg_0705_v2", file_name)
    
    mask_img = Image.fromarray(mask_np)
    mask_img.save(save_path)
    