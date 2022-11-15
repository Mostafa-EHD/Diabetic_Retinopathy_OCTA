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
#import monai
from PIL import Image, ImageOps


import torch.optim as optim
import random
import timm

### 设置参数
images_file = '../B. Image Quality Assessment/1. Original Images/a. Training Set'  # 训练图像路径

image_size = 1024 # 输入图像统一尺寸
batch_size = 4 # 批大小.
num_workers = 8 # 数据加载处理器个数

summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)



def data_split_crossval(label_file,fold= 0 ,aff_info = True):
    #fold = 0
    #label_file = 'task2_data_fold.xls'
    dataframe = pd.read_excel(label_file)
    fold0_patient = dataframe['fold0_patient'].tolist()
    fold0_label = dataframe['fold0_label'].tolist()
    fold1_patient = dataframe['fold1_patient'].tolist()
    fold1_label = dataframe['fold1_label'].tolist()
    fold2_patient = dataframe['fold2_patient'].tolist()
    fold2_label = dataframe['fold2_label'].tolist()
    fold3_patient = dataframe['fold3_patient'].tolist()
    fold3_label = dataframe['fold3_label'].tolist()
    fold4_patient = dataframe['fold4_patient'].tolist()
    fold4_label = dataframe['fold4_label'].tolist()
    fold4_patient = [x for x in fold4_patient if pd.isnull(x) == False]
    fold4_label = [int(x) for x in fold4_label if pd.isnull(x) == False]

    if fold == 0:
        train_patient = fold1_patient + fold2_patient + fold3_patient + fold4_patient
        train_label = fold1_label + fold2_label + fold3_label + fold4_label
        val_patient =  fold0_patient
        val_label = fold0_label

    if fold == 1:
        train_patient = fold0_patient + fold2_patient + fold3_patient + fold4_patient
        train_label = fold0_label + fold2_label + fold3_label + fold4_label
        val_patient =  fold1_patient
        val_label = fold1_label

    if fold == 2:
        train_patient = fold1_patient + fold0_patient + fold3_patient + fold4_patient
        train_label = fold1_label + fold0_label + fold3_label + fold4_label
        val_patient =  fold2_patient
        val_label = fold2_label

    if fold == 3:
        train_patient = fold1_patient + fold2_patient + fold0_patient + fold4_patient
        train_label = fold1_label + fold2_label + fold0_label + fold4_label
        val_patient =  fold3_patient
        val_label = fold3_label

    if fold == 4:
        train_patient = fold1_patient + fold2_patient + fold3_patient + fold0_patient
        train_label = fold1_label + fold2_label + fold3_label + fold0_label
        val_patient =  fold4_patient
        val_label = fold4_label
    if aff_info == True:
        print("Total Nums: {}, train/val: {}, test: {}".format(len(train_patient+ val_patient), len(train_patient), len(val_patient)))
        print('Train and Val patient = ', train_patient)
        print('Train and Val label = ',train_label)
        print('Test patient = ', val_patient)
        print('Test label =',val_label)
    
    return train_patient,train_label,val_patient,val_label
 

trainval_patient,trainval_label,test_patient,test_label = data_split_crossval(label_file = '../task2_data_fold.xls',fold= 4 ,aff_info = True)

# 训练/验证数据集划分
val_ratio = 0.1  # 训练/验证图像划分比例
train_patient,val_patient,train_label,val_label = train_test_split(trainval_patient,trainval_label, test_size = val_ratio, random_state=42)
print("Total Nums: {}, train: {}, val: {}".format(len(trainval_label), len(train_patient), len(val_patient)))
print('Trainset = ',train_patient)
print('Train label = ', train_label)
print('Valset = ',val_patient)
print('Val label = ', val_label)

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
            return img, label

        if self.mode == "train" or self.mode == "val" :           
            return img, label

    def __len__(self):
        return len(self.patient_list)
        
model = timm.create_model('vgg19', pretrained=True, num_classes=3, in_chans=1)

x=torch.randn(1,1,1024,1024)
output = model(x)
print(output.shape)

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.99)
criterion = nn.CrossEntropyLoss()

train_dataset = DRAC_sub2_dataset(dataset_root=images_file, 
                            patient_list = train_patient,
                            label_list = train_label,
                            mode = 'train')

val_dataset = DRAC_sub2_dataset(dataset_root=images_file, 
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
            
            for p, l in zip(logits.detach().cpu().numpy().argmax(1), labels.detach().cpu().numpy()):
                avg_kappa_list.append([p, l])
            #print(logits)
            loss = criterion(logits, labels)
            #print(loss)
            pred = logits.argmax(dim=1)
            #print(pred)
            #print(labels)
            num_correct += torch.eq(pred, labels).sum().float().item()
            #print(num_correct)
            #print(abc)

            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
                
            avg_loss_list.append(loss.item())

        avg_loss = np.array(avg_loss_list).mean()
        avg_kappa_list = np.array(avg_kappa_list)
        avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
        avg_kappa_list = []
        print("[TRAIN] epoch={}/{} avg_loss={:.4f} avg_acc={:.4f} avg_kappa={:.4f}".format(epoch, num_epochs, avg_loss, num_correct/len(train_loader.dataset), avg_kappa))
        summaryWriter.add_scalars('loss', {"loss": (avg_loss)}, epoch)
        summaryWriter.add_scalars('acc', {"acc": num_correct/len(train_loader.dataset)}, epoch)
        summaryWriter.add_scalars('kappa', {"kappa": avg_kappa}, epoch)
    
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
            pred = logits.argmax(dim=1)
            num_correct_val += torch.eq(pred, labels).sum().float().item()
            for p, l in zip(logits.detach().cpu().numpy().argmax(1), labels.detach().cpu().numpy()):
                cache.append([p, l])
            
        cache = np.array(cache)
        kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
        cache = []
        val_acc = num_correct_val/len(val_loader.dataset)
        print("[EVAL] epoch={}/{}  val_acc={:.4f} val_kappa={:.4f}".format(epoch, num_epochs, val_acc, kappa))
        summaryWriter.add_scalars('val_acc', {"val_acc": val_acc}, epoch)
        summaryWriter.add_scalars('kappa', {"val_kappa": kappa}, epoch)
        
    scheduler.step()
    filepath = './weights'
    folder = os.path.exists(filepath)
    if not folder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)
    if kappa >= best_kappa:
        print('best model epoch = ',epoch)
        print('best kappa  =',kappa)
        best_kappa = kappa
        torch.save(model.state_dict(), best_model_path) 
        
    if kappa>0.75:
        path = './weights/model' + str(epoch) + '_'+ str(kappa) + '.pth'
        torch.save(model.state_dict(), path)     

print('################# TRAIN FINISH, START TEST #################')
model = timm.create_model('vgg19', pretrained=True, num_classes=3, in_chans=1)
model.load_state_dict(torch.load(best_model_path))
model.cuda()
model.eval()

test_dataset = DRAC_sub2_dataset(dataset_root=images_file, 
                            patient_list = test_patient,
                            label_list = test_label,
                            mode = 'test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8,
                        pin_memory=True)

cache = []
num_correct_val = 0
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        #print(batch_idx)

        img = (data[0])
        labels = (data[1])            
        img = img.cuda().float()
        labels = labels.cuda()
        logits = model(img)
        pred = logits.argmax(dim=1)
        num_correct_val += torch.eq(pred, labels).sum().float().item()
        for p, l in zip(logits.detach().cpu().numpy().argmax(1), labels.detach().cpu().numpy()):
            cache.append([p, l])

    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    val_acc = num_correct_val/len(test_loader.dataset)
    print("[TEST] test_acc={:.4f} test_kappa={:.4f}".format(val_acc, kappa))


