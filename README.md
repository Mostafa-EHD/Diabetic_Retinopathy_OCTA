# Diabetic_Retinopathy_OCTA


Here's the [LaTIM](https://latim.univ-brest.fr/) team's solution in the [DRAC 2022 challenge](https://drac22.grand-challenge.org/Description/).

We are the **TOP 5** of the segmentation task, the **TOP 4** of the quality assessment task, and the **TOP 3** of the DR grading task. 

If you use our code please cite: [Segmentation, Classification, and Quality Assessment of UW-OCTA Images for the Diagnosis of Diabetic Retinopathy]().

--- 

## 1.Task 1 - Segmentation

2 techniques used to solve the segmentation problem: nnU-Net and V-Net.

![image](https://user-images.githubusercontent.com/55517267/201966441-27703cad-bac2-422a-a7ee-07bf449e715c.png)

### 1.1 nnUNet Training and Testing Tutorial for DRAC

nnUNet basic tutorial: [[pytorch] nnUnet for 2D Images Segmentation](https://blog.csdn.net/qq_38736504/article/details/125494224#t0)

The following is an example of using nnUNet for label1 segmentation, please note that we need to do the same for label2 and label3 to get the segmentation results of all three labels.

#### data preparation

Use ***1-preprocessing_nnUnet.py*** file to convert the image data to nii format, and then use the command `nnUNet_plan_and_preprocess -t 622 -pl3d None` to perform image preprocessing for nnUNet.

#### training

Five-fold cross-validation using:    
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 0 --npz`    
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 1 --npz`   
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 2 --npz`   
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 3 --npz`   
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 4 --npz`   

#### inference

Run inference on the test set using:    
`nnUNet_predict -i /data_GPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/imagesTs -o /data_GPU/yihao/GOALS/drac/622 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task622_drac`

And use ***2-nii_to_png.py*** file to convert the inference results in nii form to png format

### 1.2 V-Net

Use ***V-Net_label1.py*** and ***V-Net_label3.py*** to segment label1 and label3 data    

To alleviate over-segmentation of label3, we use ***Classifier_label3.py*** to predict the probability that an image contains label 3.

## 2. Task 2&3 - Quality assessment & Classification of DR Training

### 2.1 5-fold_cross-validation_training

To verify the performance of the different models, we used five-fold cross-validation to test six architectures (17 backbones): ResNet, DenseNet, EfficientNet, VGG, ConvNeXt, Swin-Transformer.   

***task2_data_fold.xls*** and ***task3_data_fold.xls*** are our 5-fold cross-validation split datasets. Files ***Task2_vgg19*** and ***Task3_efficientnet_b2*** are two examples of our backbone training.    

In order to achieve training/testing of different backbones, we only need to change the code of the model part.

```
model = timm.create_model('vgg19', pretrained=True, num_classes=3, in_chans=1)
```
```
backbone list = 'resnet50', 'resnet101', 'resnet152', 'resnet200d', 'densenet121', 'densenet161', 'densenet169',
'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
'vgg11', 'vgg16', 'vgg19', 'vgg13', 'convnext_base', 'swin base patch4 window7 224'
```

### 2.2 Pseudo_label_learning

Pseudo-labeling is a process that consists in adding confidently predicted test data to the training data and retraining the models.

![image](https://user-images.githubusercontent.com/55517267/201973763-09316d28-08b2-49d7-b8de-8bbf7ab59202.png)

We hard-vote the hard-to-classify data based on the results of 5-fold_cross-validation_training, add the generated pseudo-labels and high-confidence data to the original training dataset, and then use ***/Pseudo_label_learning/Task2/Task2_vgg19.py*** and ***/Pseudo_label_learning/Task3/Task3_densenet121.py + Task3_efficientnet_b3.py*** for pseudo-label learning on the new training set.

## 3. Task 2&3 - Inference

For the model weights for task two and three, you can download them at the following links: [model weights](https://drive.google.com/drive/folders/1p7-65yVdulbRUwMH5FQPF7qLatfwsMQx?usp=sharing)

### 3.1 Task2 
```
best_model_path =  './vgg19.pth'
model = timm.create_model('vgg19', pretrained=True, num_classes=3, in_chans=1)
model.load_state_dict(torch.load(best_model_path))
```

### 3.2 Task3

model ensemble
```
best_model_path =  '/densenet121.pth'
model = timm.create_model('densenet121', pretrained=True, num_classes=3, in_chans=1)

best_model_path1 =  '/efficientnet_b3.pth'
model1 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=3, in_chans=1)

....

for img, idx in _val:
    img = img.unsqueeze(0).float()
    img = img.cuda()
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

```
