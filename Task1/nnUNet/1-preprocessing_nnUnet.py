import os
import random
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib


base_image = '/home/liyihao/LI/DRAC2022/A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/1. Original Images/a. Training Set'
label1_path = '/home/liyihao/LI/DRAC2022/A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/2. Groundtruths/a. Training Set/label1'


filelists = os.listdir(label1_path)
print(len(filelists))
print(filelists)


### train set - imagesTr ###
for name in filelists:
    num = name.split('.')[0]
    print(num)
    volume = np.zeros((1024,1024,1))
    volume[:,:,0] = cv2.imread(os.path.join(base_image, name),cv2.IMREAD_GRAYSCALE) 
    
    print(volume.shape)
    new_image = nib.Nifti1Image(volume, affine=np.eye(4))
    path = '/home/liyihao/ClusterGPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/imagesTr/drac_'+num+'_0000.nii.gz'
    nib.save(new_image, path)


### train set - labelsTr ###
for name in filelists:
    num = name.split('.')[0]
    print(num)
    volume = np.zeros((1024,1024,1))
    if name in os.listdir(label1_path):
        img = cv2.imread(os.path.join(label1_path, name),cv2.IMREAD_GRAYSCALE)
        img[img==255]=1
        volume[:,:,0] = img


    new_image = nib.Nifti1Image(volume, affine=np.eye(4))
    path = '/home/liyihao/ClusterGPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/labelsTr/drac_'+num+'.nii.gz'
    nib.save(new_image, path)

### test set - imagesTs ###
label1_path = '/home/liyihao/LI/DRAC2022/DRAC2022_Testing_Set/A. Segmentation/1. Original Images/b. Testing Set'
base_image = label1_path

filelists = os.listdir(label1_path)
print(len(filelists))
print(filelists)

for name in filelists:
    num = name.split('.')[0]
    print(num)
    volume = np.zeros((1024,1024,1))
    volume[:,:,0] = cv2.imread(os.path.join(base_image, name),cv2.IMREAD_GRAYSCALE) 
    
    print(volume.shape)
    new_image = nib.Nifti1Image(volume, affine=np.eye(4))
    #path = '/home/liyihao/ClusterGPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/imagesTr/drac_'+num+'_0000.nii.gz'
    path = '/home/liyihao/ClusterGPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/imagesTs/drac_'+num+'_0000.nii.gz'
    nib.save(new_image, path)
    
### json ###
import glob
import os
import re
import json
from collections import OrderedDict

filelists = os.listdir('/home/liyihao/ClusterGPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/labelsTr')
print(len(filelists))

json_dict = OrderedDict()
json_dict['name'] = "drac2022"
json_dict['description'] = "LI Yihao copyright"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "see drac2022"
json_dict['licence'] = "see drac2022"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "OCTA"
}
json_dict['labels'] = {
    "0": "background",
    "1": "label"
}

json_dict['numTraining'] = len(filelists)
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in
                         filelists]
json_dict['test'] = []

with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(json_dict, f, ensure_ascii=False, indent=4)


