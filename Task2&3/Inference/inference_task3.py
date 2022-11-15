import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.model_selection import train_test_split
import monai
from PIL import Image, ImageOps


import torch.optim as optim
import random
import timm

### 设置参数
images_file = '/home/liyihao/LI/DRAC2022/DRAC2022_Testing_Set/C. Diabetic Retinopathy Grading/1. Original Images/b. Testing Set'  # 训练图像路径
#images_file = '/home/liyihao/LI/DRAC2022/DRAC2022_Testing_Set/B. Image Quality Assessment/1. Original Images/b. Testing Set'
image_size = 1024 # 输入图像统一尺寸

batch_size = 4 # 批大小.
num_workers = 8 # 数据加载处理器个数


patient_list = os.listdir(images_file)
print(len(patient_list))
print(patient_list)

# 数据加载
class DRAC_sub2_dataset(Dataset):
    def __init__(self,
                dataset_root,
                patient_list ='',
                label_list ='',
                mode='train'):
        self.dataset_root = dataset_root
        self.mode = mode
        self.patient_list = patient_list
        self.label_list = label_list
    
    def __getitem__(self, idx):
        
        patient_name = self.patient_list[idx]
        if self.mode == "train" or self.mode == "val" : 
            label = self.label_list[idx]
        
        img_path = os.path.join(self.dataset_root, patient_name)    
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = img.resize((image_size,image_size))
        # normlize on GPU to save CPU Memory and IO consuming.
        # img = (img / 255.).astype("float32")
        
        if self.mode == "train":
            im_aug = transforms.Compose([
                #tfs.Resize(120),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                #transforms.RandomPerspective(),
                #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
                #transforms.RandomInvert(),
                #transforms.RandomPosterize(bits=2),
                #transforms.RandomAdjustSharpness(sharpness_factor=2),
                #transforms.RandomAutocontrast(),
                #transforms.RandomEqualize()
            ])
            img = im_aug(img)
        
        img = transforms.PILToTensor()(img)
        #print(img.shape)

        #img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

        if self.mode == 'test':
            return img, patient_name

        if self.mode == "train" or self.mode == "val" :           
            return img, label

    def __len__(self):
        return len(self.patient_list)
        
_val = DRAC_sub2_dataset(dataset_root=images_file, 
                          patient_list = patient_list,
                          label_list = [],
                          mode = 'test')

for i in range(20):
    img, name = _val.__getitem__(i)
    img = img.numpy()
    print(img.shape)
    print(name)

best_model_path =  '/densenet121.pth'
model = timm.create_model('densenet121', pretrained=True, num_classes=3, in_chans=1)
model.load_state_dict(torch.load(best_model_path))
model.cuda()
model.eval()

best_model_path1 =  '/efficientnet_b3.pth'
model1 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=3, in_chans=1)
model1.load_state_dict(torch.load(best_model_path1))
model1.cuda()
model1.eval()

case = []
class_list = []
p0 = []
p1 = []
p2 = []

for img, idx in _val:
    img = img.unsqueeze(0).float()
    img = img.cuda()
    #print(img.shape)
    print(idx)
    #print(abc)
    #img = img[np.newaxis, ...]
    #img = paddle.to_tensor((img / 255.).astype("float32"))
    logits = model(img)
    logits1 = model1(img)
    logits_ensm = 0.5*logits + 0.5*logits1
    m = nn.Softmax()
    pred = m(logits_ensm).detach().cpu().numpy()
    case.append(idx)
    class_list.append(pred[0].argmax())
    p0.append(pred[0][0])
    p1.append(pred[0][1])
    p2.append(pred[0][2])
    
    
    
from pandas.core.frame import DataFrame
c={"case":case,
   "class":class_list,
   "P0":p0,
   "P1":p1,
   "P2":p2}
data = DataFrame(c)
print(data)
data.to_csv('/home/liyihao/LI/DRAC2022/task3.csv',index=False)



