# nnUNet Training and Testing Tutorial for DRAC

nnUNet basic tutorial: [[pytorch] nnUnet for 2D Images Segmentation](https://blog.csdn.net/qq_38736504/article/details/125494224#t0)

The following is an example of using nnUNet for label1 segmentation, please note that we need to do the same for label2 and label3 to get the segmentation results of all three labels.

## data preparation

Use 1-preprocessing_nnUnet.py file to convert the image data to nii format, and then use the command `nnUNet_plan_and_preprocess -t 622 -pl3d None` to perform image preprocessing for nnUNet.

## training

Five-fold cross-validation using:    
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 0 --npz`    
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 1 --npz`   
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 2 --npz`   
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 3 --npz`   
`nnUNet_train 2d nnUNetTrainerV2 Task622_drac 4 --npz`   

## inference

Run inference on the test set using:    
`nnUNet_predict -i /data_GPU/yihao/GOALS/nnUnet/DATASET/nnUNet_raw/nnUNet_raw_data/Task622_drac/imagesTs -o /data_GPU/yihao/GOALS/drac/622 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task622_drac`

And use 2-nii_to_png.py file to convert the inference results in nii form to png format
