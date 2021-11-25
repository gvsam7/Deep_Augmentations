# Deep_Augmentations

## Description

This project is part of the British academy [Satellite/Aerial Image Segmentation](https://wearepal.ai/projects/ssrp) project. 
This is a test platform to validate the impact on model robustness of various data augmentation techniques on various
Deep architectures using Sussex Sustainability Research Programme (SSRP) dataset. The dataset is comprised of Satellite/Aerial Images from Ghaziabad India.
Stratification method was used to split the data to train/validate: 80% (out of which train: 80% and
validation: 20%), and test: 20% data.
Architectures: 4, and 5 CNN, ResNet18, ResNet50, ResNet101, ResNet152, VGG11, VGG13, VGG16, VGG19, AlexNet.
Images: compressed 50x50, 100x100, and 226x226 pixel images. Note that 50x50 was too small for the 5 CNNs.
Test Procedure: 5 runs for each architecture for each of the compressed data. Then the Interquartile range, using the median was plotted.
Plots: Average GPU usage per architecture, Interquartile, and for each architecture an F1 Score heatmap
for each class.
Data augmentation techniques tested: Cutout, mixup, CutMix and AugMix

![alt text](https://github.com/gvsam7/Deep_Augmentations/blob/main/Images/SSRP_Classifier.PNG)


## Papers
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)