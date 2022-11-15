import os
import random
from tqdm import tqdm
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_dir = '/home/liyihao/ClusterGPU/yihao/GOALS/drac/622'

img_list = [i for i in os.listdir(img_dir) if ".nii.gz" in i]
print(img_list)
print(len(img_list))


with tqdm(img_list, desc="conver") as pbar:
    for name in pbar:
        print(name)
        image = sitk.ReadImage(os.path.join(img_dir, name))
        image = sitk.GetArrayFromImage(image)[0]
        #image[image == 1] = 255
        print(image.shape)
        print(np.unique(image))
        image = image.transpose((1,0))
        cv2.imwrite(os.path.join('/home/liyihao/LI/DRAC2022/label1_v5', name.split(".")[0].split("_")[1]+".png"), image)
        
        
