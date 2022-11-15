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
from PIL import Image
from monai.losses.dice import DiceLoss
from torchvision.transforms.functional import to_pil_image,affine
from monai.transforms import Rand2DElastic
### 设置参数
images_file = '../A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/1. Original Images/a. Training Set'  # 训练图像路径
gt_file = '../A. Segmentation_WILL_RELEASE_SOON/A. Segmentation/2. Groundtruths/a. Training Set/1. Intraretinal Microvascular Abnormalities'
image_size = 1024 # 输入图像统一尺寸
val_ratio = 0.1  # 训练/验证图像划分比例
batch_size = 3 # 批大小
iters = 10000 # 训练迭代次数
optimizer_type = 'adam' # 优化器, 可自行使用其他优化器，如SGD, RMSprop,...
num_workers = 8 # 数据加载处理器个数
init_lr = 1e-3 # 初始学习率



summary_dir = './logs'
torch.backends.cudnn.benchmark = True
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
summaryWriter = SummaryWriter(summary_dir)


# 训练/验证数据集划分
filelists = os.listdir(gt_file)
print(filelists)
train_filelists, val_filelists = train_test_split(filelists, test_size = val_ratio)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

### 从数据文件夹中加载眼底图像，提取相应的金标准，生成训练样本
class OCTDataset(Dataset):
    def __init__(self, image_file, gt_path=None, filelists=None,  mode='train'):
        super(OCTDataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        image_idxs = os.listdir(gt_path) # 0001.png,
        self.gt_path = gt_path
        self.file_list = [image_idxs[i] for i in range(len(image_idxs))]        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item in filelists] 
    
    def transform(self,img, mask):
        (d, t, sc, sh) = transforms.RandomAffine.get_params(degrees=(-40, 40), translate=(0.4, 0.4),
                                                            scale_ranges=(0.6, 1.4), shears=(-40, 40),
                                                            img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask = affine(to_pil_image(mask), angle=d, translate=t, scale=sc, shear=sh)

        return (np.array(img), np.array(mask))
   
    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        img_path = os.path.join(self.image_path, real_index)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
        #img = img[:,:,np.newaxis]
        #print(img.shape)
        #h,w = img.shape # (800, 1100, 3)     
        img = cv2.resize(img,(image_size, image_size))
        #img = img[:,:,np.newaxis]
        #print(img.shape)
        
        if self.mode == 'train' or self.mode == 'val':
            gt_tmp_path = os.path.join(self.gt_path, real_index)
            gt_img = cv2.imread(gt_tmp_path,cv2.IMREAD_GRAYSCALE)

            ### 像素值为0的是RNFL(类别 0)，像素值为80的是GCIPL(类别 1)，像素值为160的是脉络膜(类别 2)，像素值为255的是其他（类别3）。
            gt_img[gt_img == 255] = 1
            
            gt_img = cv2.resize(gt_img,(image_size, image_size),interpolation = cv2.INTER_NEAREST)
            #gt_img = gt_img[:,:,1]
            #print('gt shape', gt_img.shape)           
        
        if self.mode == 'train':
            img, gt_img = self.transform(img, gt_img)            
            deform = Rand2DElastic(
                prob=0.5,
                spacing=(50, 50),
                magnitude_range=(5, 6),
                rotate_range=(np.pi / 4,),
                scale_range=(0.2, 0.2),
                translate_range=(100, 100),
                padding_mode="zeros",
                device="cpu")
            
            deform.set_random_state(seed=23)
            img = deform(img[np.newaxis,:,:], (image_size, image_size), mode="bilinear")
            deform.set_random_state(seed=23)
            gt_img = deform(gt_img[np.newaxis,:,:], (image_size, image_size), mode="nearest")
        
        if self.mode == 'train' or self.mode == 'val':
            #gt_img = gt_img[:,:,np.newaxis]
            #gt_img = gt_img.transpose(2,0,1)
            if gt_img.shape[0] != 1:
                gt_img = gt_img[np.newaxis,:,:]
            gt_img = torch.from_numpy(gt_img)
            
        #print(img.shape)
        if img.shape[0] != 1: 
            img = img[np.newaxis,:,:]
        img = torch.from_numpy(img)
        
        
        # print(img.shape)
        # img = img_re.astype(np.float32)
        
        
        if self.mode == 'test':
            ### 在测试过程中，加载数据返回眼底图像，数据名称，原始图像的高度和宽度
            return img, real_index
        
        if self.mode == 'train' or self.mode == 'val':
            ###在训练过程中，加载数据返回眼底图像及其相应的金标准           
            return img, gt_img

    def __len__(self):
        return len(self.file_list)
        
model = monai.networks.nets.VNet(in_channels=1, out_channels=2,spatial_dims=2)
best_model_path =  '../Task1_vnet_label1_v4/weights/model20_0.49956390256490507.pth'
model.load_state_dict(torch.load(best_model_path))
model.cuda()

metric = DiceLoss(to_onehot_y = True, softmax = True, include_background = False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.99)

train_dataset = OCTDataset(image_file = images_file, 
                        gt_path = gt_file,
                        filelists=train_filelists)

val_dataset = OCTDataset(image_file = images_file, 
                        gt_path = gt_file,
                        filelists=val_filelists,mode='val')


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        pin_memory=True)

def get_dice(gt, pred, classId=1):
    if np.sum(gt) == 0:
        return np.nan
    else:
        intersection = np.logical_and(gt == classId, pred == classId)
        dice_eff = (2. * intersection.sum()) / (gt.sum() + pred.sum())
        return dice_eff


def get_IoU(gt, pred, classId=1):
    if np.sum(gt) == 0:
        return np.nan
    else:
        intersection = np.logical_and(gt == classId, pred == classId)
        union = np.logical_or(gt == classId, pred == classId)
        iou = np.sum(intersection) / np.sum(union)
        return iou


def get_mean_IoU_dice(gts_list, preds_list):
    assert len(gts_list) == len(preds_list)
    dice_list = []
    iou_list = []
    for gt_array, pred_array in zip(gts_list, preds_list):
        dice = get_dice(gt_array, pred_array, 1)
        iou = get_IoU(gt_array, pred_array, 1)
        dice_list.append(dice)
        iou_list.append(iou)
    mDice = np.nanmean(dice_list)
    mIoU = np.nanmean(iou_list)
    return mDice, mIoU

num_epochs = 1000
for epoch in range(num_epochs):
    #print('lr now = ', get_learning_rate(optimizer))
    avg_loss_list = []
    avg_dice_list = []
    
    model.train()
    with torch.enable_grad():
        for batch_idx, data in enumerate(train_loader):
            img = (data[0]).float()
            gt_label = (data[1])
            #print(img.shape)
            #print(gt_label.shape)
            
            img = img.cuda()
            gt_label = gt_label.cuda()
            
            
            logits = model(img)
            #print(logits)
            dice = metric(logits,gt_label)
            #loss = criterion(logits, torch.squeeze(gt_label,dim=1).long()) + dice
            loss = dice
            #print(loss)
            
            avg_loss_list.append(loss.item())
            avg_dice_list.append(dice.item())
            

            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None
            
        avg_loss = np.array(avg_loss_list).mean()
        avg_dice = np.array(avg_dice_list).mean()
        print("[TRAIN] epoch={}/{} avg_loss={:.4f} avg_dice={:.4f}".format(epoch, num_epochs, avg_loss, avg_dice))
        summaryWriter.add_scalars('loss', {"loss": (avg_loss)}, epoch)
        summaryWriter.add_scalars('dice', {"dice": avg_dice}, epoch)
        
    model.eval()
    pred_img_list = []
    gt_label_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            
            img = (data[0]).float()
            gt_label = (data[1])

            img = img.cuda()
            gt_label = gt_label.numpy()


            logits = model(img)
            
            pred_img = logits.detach().cpu().numpy().argmax(1).squeeze()
            gt_label = np.squeeze(gt_label)
            
            pred_img_list.append(pred_img)
            gt_label_list.append(gt_label)
            #print(np.unique(pred_img))
            
            
#             mean_Dice, mean_IoU = get_mean_IoU_dice(gt_label_list, pred_img_list)
#             print(mean_Dice)
#             print(mean_IoU)
#             print(pred_img.shape)
#             print(gt_label.shape)
#             print(abc)

        mean_Dice, mean_IoU = get_mean_IoU_dice(gt_label_list, pred_img_list)
        print("[EVAL] epoch={}/{}  mean_Dice={:.4f} mean_IoU={:.4f} ".format(epoch, num_epochs,mean_Dice,mean_IoU))
        summaryWriter.add_scalars('mean_Dice', {"mean_Dice": mean_Dice}, epoch)
        summaryWriter.add_scalars('mean_IoU', {"mean_IoU": mean_IoU}, epoch)
        
    scheduler.step()

    filepath = './weights'
    folder = os.path.exists(filepath)
    if not folder:
        # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(filepath)
    path = './weights/model' + str(epoch) + '_'+ str(mean_Dice) + '.pth'
    torch.save(model.state_dict(), path)            

summaryWriter.close()

  


