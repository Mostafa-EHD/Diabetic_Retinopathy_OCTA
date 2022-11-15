import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
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
from sklearn import metrics

import torch.optim as optim
import random
import timm

### 设置参数
images_file = './A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/1. Original Images/a. Training Set'  # 训练图像路径
#images_file = './C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set'
image_size = 1024 # 输入图像统一尺寸

batch_size = 4 # 批大小.
num_workers = 8 # 数据加载处理器个数


label1_path = './A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/2. Groundtruths/a. Training Set/1. Intraretinal Microvascular Abnormalities'
label2_path = './A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/2. Groundtruths/a. Training Set/2. Nonperfusion Areas'
label3_path = './A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/2. Groundtruths/a. Training Set/3. Neovascularization'

summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)

filelists = os.listdir(images_file)
print(len(filelists))

label_list = []
patient_list = []
for name in filelists:
    label = []
    if name in os.listdir(label1_path):
        label.append(1)
    else:
        label.append(0)
    if name in os.listdir(label2_path):
        label.append(1)
    else:
        label.append(0)
    if name in os.listdir(label3_path):
        label.append(1)
    else:
        label.append(0)
    patient_list.append(name)
    label_list.append(label)
print(label_list)


val_ratio = 0.1  # 训练/验证图像划分比例
train_patient,val_patient,train_label,val_label = train_test_split(patient_list,label_list, test_size = val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_patient), len(val_patient)))
print(pd.value_counts(val_label))


# 数据加载
class DRAC_sub1_dataset(Dataset):
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
        label = torch.FloatTensor(label)
        
        #print(img.shape)

        #img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

        if self.mode == 'test':
            return img, label

        if self.mode == "train" or self.mode == "val" :           
            return img, label

    def __len__(self):
        return len(self.patient_list)
        
        
model = timm.create_model('resnet101', pretrained=True, num_classes=3, in_chans=1)


model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.99)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

train_dataset = DRAC_sub1_dataset(dataset_root=images_file, 
                            patient_list = train_patient,
                            label_list = train_label,
                            mode = 'train')

val_dataset = DRAC_sub1_dataset(dataset_root=images_file, 
                          patient_list = val_patient,
                          label_list = val_label,
                          mode = 'val')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8,
                        pin_memory=True)
                        
num_epochs = 1000
best_kappa = 0.0
best_model_path = './weights/bestmodel.pth'
for epoch in range(num_epochs):
    #print('lr now = ', get_learning_rate(optimizer))
    avg_loss_list = []
    num_correct = 0
    avg_kappa_list = []
    label1_pre = []
    label2_pre = []
    label3_pre = []
    label1_gt = []
    label2_gt = []
    label3_gt = []
    
    
    model.train()
    with torch.enable_grad():
        for batch_idx, data in enumerate(train_loader):
            
            img = (data[0])
            labels = (data[1])
            
            img = img.cuda().float()
            labels = labels.cuda()
            
            
            #print(img.shape)
            #print(labels)
            

            logits = model(img)
            loss = criterion(logits, labels)
            
            m = nn.Sigmoid()
            output = m(logits).detach().cpu().numpy()
            gt = labels.detach().cpu().numpy()
            
            
            
            
            for i in range(img.shape[0]):
                label1_pre.append(output[i][0])
                label2_pre.append(output[i][1])
                label3_pre.append(output[i][2])
                label1_gt.append(gt[i][0])
                label2_gt.append(gt[i][1])
                label3_gt.append(gt[i][2])
                
            
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
                
            avg_loss_list.append(loss.item())
            
#             print(m(logits))
#             print(label1_gt)
#             print(label1_pre)
#             print(label2_gt)
#             print(label2_pre)
#             print(loss)
        avg_loss = np.array(avg_loss_list).mean()
        #print(label1_gt)

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label1_gt, label1_pre)
        auc1 = sklearn.metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label2_gt, label2_pre)
        auc2 = sklearn.metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label3_gt, label3_pre)
        auc3 = sklearn.metrics.auc(fpr, tpr)
        
        print("[TRAIN] epoch={}/{} avg_loss={:.4f} auc_avg ={:.4f} auc1 ={:.4f} auc3 ={:.4f}".format(epoch, num_epochs, avg_loss, (auc1+auc2+auc3)/3,auc1,auc3))
        summaryWriter.add_scalars('loss', {"loss": (avg_loss)}, epoch)
        summaryWriter.add_scalars('auc1', {"auc1": auc1}, epoch)
        summaryWriter.add_scalars('auc1', {"auc2": auc2}, epoch)
        summaryWriter.add_scalars('auc1', {"auc3": auc3}, epoch)
        summaryWriter.add_scalars('auc_avg', {"auc_avg": (auc1+auc2+auc3)/3}, epoch)
        
    label1_pre = []
    label2_pre = []
    label3_pre = []
    label1_gt = []
    label2_gt = []
    label3_gt = []
    model.eval()
    cache = []
    num_correct_val = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            
            img = (data[0])
            labels = (data[1])            
            img = img.cuda().float()
            labels = labels.cuda()
            logits = model(img)
            
            m = nn.Sigmoid()
            output = m(logits).detach().cpu().numpy()
            gt = labels.detach().cpu().numpy()
            
            for i in range(img.shape[0]):
                label1_pre.append(output[i][0])
                label2_pre.append(output[i][1])
                label3_pre.append(output[i][2])
                label1_gt.append(gt[i][0])
                label2_gt.append(gt[i][1])
                label3_gt.append(gt[i][2])
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(label1_gt, label1_pre)
                
        auc1 = sklearn.metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label2_gt, label2_pre)
        auc2 = sklearn.metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label3_gt, label3_pre)
        auc3 = sklearn.metrics.auc(fpr, tpr)
        
        print("[VAL] epoch={}/{} auc_avg ={:.4f} auc1 ={:.4f} auc3 ={:.4f}".format(epoch, num_epochs, (auc1+auc3)/2,auc1,auc3))
        summaryWriter.add_scalars('auc1', {"val_auc1": auc1}, epoch)
        summaryWriter.add_scalars('auc1', {"val_auc2": auc2}, epoch)
        summaryWriter.add_scalars('auc1', {"val_auc3": auc3}, epoch)
        summaryWriter.add_scalars('auc_avg', {"val_auc_avg": (auc1+auc3)/2}, epoch)
        
    scheduler.step()
    filepath = './weights'
    folder = os.path.exists(filepath)
    if not folder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)

    path = './weights/model' + str(epoch) + '_'+ str((auc1+auc3)/2) + '.pth'
    torch.save(model.state_dict(), path)  
                        




