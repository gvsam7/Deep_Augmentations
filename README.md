# Deep_Augmentations

## Description

This project is part of the British academy [Satellite/Aerial Image Segmentation](https://wearepal.ai/projects/ssrp) project. 
The aim is to apply deep learning techniques, to map peri-urban agriculture in Ghaziabad India; and research ways of integrating multiple types of data through a web-based mapping and visualisation tool. Thus, support research and stakeholder engagement to understand the trade-offs between Sustainable Development goals (SDGs) in urbanising contexts.
For this project, a classifier that classifies scenes from aerial images was designed. The architecture is comprised of Convolutional Neural Networks. The dataset is comprised of satellite/aerial images depicting land scenes from Ghaziabad India. Classifier predictions are imported to the web application for visualisation. 

Moreover, conducting an investigation regarding Deep Learning model robustness to domain shifts such as year/season data sources.

![alt text](https://github.com/gvsam7/Deep_Augmentations/blob/main/Images/SSRP_Classifier.PNG)

*Data:*  Sussex Sustainability Research Programme (SSRP) dataset.
Stratification method was used to split the data to train/validate: 80% (out of which train: 80% and
validation: 20%), and test: 20% data.

*Data texture bias:* Research techniques that take advantage texture bias in aerial/satellite images.

*Architectures:* 4, and 5 CNN, ResNet18, ResNet50, ResNet101, ResNet152, VGG11, VGG13, VGG16, VGG19, AlexNet.

*Images:* compressed 50x50, 100x100, and 256x256 pixel images. Note that 50x50 was too small for the 5 CNNs.

*Test Procedure:* 5 runs for each architecture for each of the compressed data. Then plot the Interquartile range.

*Plots:* Average GPU usage per architecture, Interquartile, F1 Score heatmap for each class, Confusion Matrix, PCA and t-SNE plots, and most confident incorrect predictions.

*Data augmentations:* Geometric Transformations, Cutout, Mixup, and CutMix, Pooling (Global Pool, Mix Pool, Gated Mix Pool).

## Google Colab
- [CNN5 Classification Activation Map (CAM)](https://colab.research.google.com/drive/1jICBINwqq2AC9N5vtme7UfkJHTxK8yIR]
- [GaborCNN Classification Activation Map (CAM)](https://colab.research.google.com/drive/1QX8Egvzb4NTzcDK9qPkiMeq8p9eEc6sr]

## Papers
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
- [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
- [Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree](https://arxiv.org/abs/1509.08985)
- [ImageNet-Trained CNNs are Biased Towards Texture; Increasing Shape Bias Improves Accuracy and Robustness](https://arxiv.org/abs/1811.12231)
- [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](https://arxiv.org/abs/1911.09071)
- [Combining Diverse Feature Priors](https://arxiv.org/abs/2110.08220)