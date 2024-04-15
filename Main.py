"""
Author: Georgios Voulgaris
Date: 25/05/2021
Description: Apply deep learning techniques, to map peri-urban agriculture in Ghaziabad India; and research ways of
            integrating multiple types of data through a web-based mapping and visualisation tool. Classify scenes from
            aerial images. The dataset is comprised of satellite/aerial images depicting land scenes from Ghaziabad
            India. Classifier predictions are imported to the web application for visualisation.

            Data: Sussex Sustainability Research Programme (SSRP) dataset. Stratification method was used to split the
            data to train/validate: 80% (out of which train: 80% and validation: 20%), and test: 20% data.

            Data texture bias: Research techniques that take advantage texture bias in aerial/satellite images.
            Architectures: 4, and 5 CNN, ResNet18, ResNet50, ResNet101, ResNet152, VGG11, VGG13, VGG16, VGG19, AlexNet.

            Images: compressed 50x50, 100x100, and 256x256 pixel images. Note that 50x50 was too small for the 5 CNNs.

            Test Procedure: 5 runs for each architecture for each of the compressed data. Then plot the Interquartile
            range.

            Plots: Average GPU usage per architecture, Interquartile, and for each architecture an F1 Score heatmap
            for each class.

            Data augmentation: Geometric Transformations, Cutout, Mixup, and CutMix, Pooling (Global Pool, Mix Pool,
            Gated Mix Pool).
"""

# Imports
import torch
from torch import optim
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm  # For nice progress bar!
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from Utilities.Save import save_checkpoint, load_checkpoint
from Utilities.Data import DataRetrieve
from Utilities.config import train_transforms, val_transforms, test_transforms
from Utilities.Networks import networks
from Utilities.Hyperparameters import arguments
from plots.ModelExam import parameters, get_predictions, plot_confusion_matrix, plot_most_incorrect, get_representations, get_pca, plot_representations, get_tsne
from CutMix.Cutout import mask
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import wandb
import sys
import os


def step(data, targets, model, optimizer, criterion, train):
    with torch.set_grad_enabled(train):
        outputs = model(data)
        acc = outputs.argmax(dim=1).eq(targets).sum().item()
        loss = criterion(outputs, targets)

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc, loss


@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = []
    for x, _ in loader:
        x = x.to(device)
        preds = model(x)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).cpu()
    return all_preds


def main():
    args = arguments()
    # wandb.init(entity="predictive-analytics-lab", project="Augmentations", config=args)
    wandb.init(project="Augmentations", config=args)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Load Data
    if args.dataset == "bw":
        dataset = ImageFolder("BW_Training_Data_2018_2014")
        in_channels = 1
    elif args.dataset == 'bw_wet':
        traindataset = ImageFolder("BWData_WetSeason")
        testdataset = ImageFolder("BWData_DrySeason")
        in_channels = 3
    elif args.dataset == "bw_dry":
        traindataset = ImageFolder("BWData_DrySeason")
        testdataset = ImageFolder("BWData_WetSeason")
        in_channels = 3
    elif args.dataset == "edge":
        dataset = ImageFolder("Edge_Training_Data_2018_2014")
        in_channels = 1
    elif args.dataset == "edge_wet":
        traindataset = ImageFolder("EdgeData_WetSeason")
        testdataset = ImageFolder("EdgeData_DrySeason")
        in_channels = 3
    elif args.dataset == "edge_dry":
        traindataset = ImageFolder("EdgeData_DrySeason")
        testdataset = ImageFolder("EdgeData_WetSeason")
        in_channels = 3
    elif args.dataset == 'sobel':
        dataset = ImageFolder("Sobel_Training_Data_2018_2014")
        in_channels = 1
    elif args.dataset == "sobel_wet":
        traindataset = ImageFolder("SobelData_WetSeason")
        testdataset = ImageFolder("SobelData_DrySeason")
        in_channels = 3
    elif args.dataset == "sobel_dry":
        traindataset = ImageFolder("SobelData_DrySeason")
        testdataset = ImageFolder("SobelData_WetSeason")
        in_channels = 3
    elif args.dataset == 'wet':
        traindataset = ImageFolder("Data_WetSeason")
        testdataset = ImageFolder("Data_DrySeason")
        in_channels = 3
    elif args.dataset == 'dry':
        traindataset = ImageFolder("Data_DrySeason")
        testdataset = ImageFolder("Data_WetSeason")
        in_channels = 3
    elif args.dataset == 'bwsobel_dry':
        traindataset = ImageFolder("BWSobelData_DrySeason")
        testdataset = ImageFolder("Data_WetSeason")
        in_channels = 3
    elif args.dataset == 'realproduct':
        traindataset = ImageFolder("Real_World")
        testdataset = ImageFolder("Product")
        in_channels = 3
    elif args.dataset == 'productreal':
        traindataset = ImageFolder("Product")
        testdataset = ImageFolder("Real_World")
        in_channels = 3
    elif args.dataset == 'realart':
        traindataset = ImageFolder("Real_World")
        testdataset = ImageFolder("Art")
        in_channels = 3
    elif args.dataset == 'artreal':
        traindataset = ImageFolder("Art")
        testdataset = ImageFolder("Real_World")
        in_channels = 3
    elif args.dataset == 'realclipart':
        traindataset = ImageFolder("Real_World")
        testdataset = ImageFolder("Clipart")
        in_channels = 3
    elif args.dataset == 'clipartreal':
        traindataset = ImageFolder("Clipart")
        testdataset = ImageFolder("Real_World")
        in_channels = 3
    elif args.dataset == 'productclipart':
        traindataset = ImageFolder("Product")
        testdataset = ImageFolder("Clipart")
        in_channels = 3
    elif args.dataset == 'clipartproduct':
        traindataset = ImageFolder("Clipart")
        testdataset = ImageFolder("Product")
        in_channels = 3
    elif args.dataset == 'productart':
        traindataset = ImageFolder("Product")
        testdataset = ImageFolder("Art")
        in_channels = 3
    elif args.dataset =='artproduct':
        traindataset = ImageFolder("Art")
        testdataset = ImageFolder("Product")
        in_channels = 3
    elif args.dataset == 'artclipart':
        traindataset = ImageFolder("Art")
        testdataset = ImageFolder("Clipart")
        in_channels = 3
    elif args.dataset == 'clipartart':
        traindataset = ImageFolder("Clipart")
        testdataset = ImageFolder("Art")
        in_channels = 3
    elif args.dataset == 'resisc45':
        dataset = ImageFolder("resisc45")
        in_channels = 3
    elif args.dataset == 'eurosat':
        dataset = ImageFolder("eurosat")
        in_channels = 3
    else:
        dataset = ImageFolder("Training_Data_2018_2014")
        in_channels = 3
    print(f"Dataset is {args.dataset}")

    if args.dataset == 'wet' or args.dataset == 'dry' or args.dataset == 'bw_wet' or args.dataset == 'edge_wet' or \
            args.dataset == 'sobel_wet' or args.dataset == 'bw_dry' or args.dataset == 'edge_dry' or \
            args.dataset == 'sobel_dry' or args.dataset == 'realproduct' or args.dataset == 'productreal'\
            or args.dataset == 'realart' or args.dataset == 'artreal' or args.dataset == 'realclipart' \
            or args.dataset == 'clipartreal' or args.dataset == 'productclipart' or args.dataset == 'clipartproduct'\
            or args.dataset == 'productart' or args.dataset == 'artproduct' or args.dataset == 'artclipart' \
            or args.dataset == 'clipartart':
        labels = traindataset.classes
        num_classes = len(labels)
        y = traindataset.targets
        y_test = testdataset.targets
        dataset_len = len(traindataset)

        X_train, X_val, y_train, y_val = train_test_split(np.arange(dataset_len), y, test_size=0.2, stratify=y,
                                                          random_state=args.random_state, shuffle=True)
        train_ds = Subset(traindataset, X_train)
        val_ds = Subset(traindataset, X_val)
        test_ds = testdataset
        filepaths = np.array(tuple(zip(*traindataset.imgs))[0])
        train_filepaths = filepaths[X_train]
        val_filepaths = filepaths[X_val]

        # Create train, validation and test datasets
        train_dataset = DataRetrieve(
            train_ds,
            transforms=train_transforms(args.width, args.height, args.Augmentation),
            augmentations=args.augmentation
        )

        val_dataset = DataRetrieve(
            val_ds,
            transforms=val_transforms(args.width, args.height)
        )

        test_dataset = DataRetrieve(
            test_ds,
            transforms=test_transforms(args.width, args.height)
        )

        # Create train, validation and test dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        prediction_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    else:
        labels = dataset.classes
        num_classes = len(labels)
        y = dataset.targets
        dataset_len = len(dataset)

        X_trainval, X_test, y_trainval, y_test = train_test_split(np.arange(dataset_len), y, test_size=0.2, stratify=y,
                                                                  random_state=args.random_state, shuffle=True)
        X2 = X_trainval
        y2 = y_trainval
        X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2, stratify=y2,
                                                          random_state=args.random_state, shuffle=True)
        train_ds = Subset(dataset, X_train)
        val_ds = Subset(dataset, X_val)
        test_ds = Subset(dataset, X_test)
        filepaths = np.array(tuple(zip(*dataset.imgs))[0])
        train_filepaths = filepaths[X_train]
        val_filepaths = filepaths[X_val]
        test_filepaths = filepaths[X_test]

        # Create train, validation and test datasets
        train_dataset = DataRetrieve(
            train_ds,
            transforms=train_transforms(args.width, args.height, args.Augmentation),
            augmentations=args.augmentation
        )

        val_dataset = DataRetrieve(
            val_ds,
            transforms=val_transforms(args.width, args.height)
        )

        test_dataset = DataRetrieve(
            test_ds,
            transforms=test_transforms(args.width, args.height)
        )

        # Create train, validation and test dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        prediction_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Network
    model = networks(architecture=args.architecture, in_channels=args.in_channels, num_classes=num_classes,
                     pretrained=args.pretrained, requires_grad=args.requires_grad, global_pooling=args.global_pooling).to(device)
    print(model)
    n_parameters = parameters(model)
    print(f"The model has {n_parameters:,} trainable parameters")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load model
    if args.load_model == 'True':
        print(f"Load model is {args.load_model}")
        if device == torch.device("cpu"):
            load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), model, optimizer)
        else:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Train Network
    for epoch in range(args.epochs):
        model.train()
        sum_acc = 0
        for data, targets in train_loader:
            # for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            if args.augmentation == "cutmix":
                None
            else:
                acc, loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=True)
                sum_acc += acc

            """
            # Visualise Cutout
            # Denormalise images so they will not be dark
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])

            data = data * std[:, None, None] + mean[:, None, None]
            
            print(f"Data_shape: {data.shape}")
            print("Original images: ")
            fig = plt.figure(figsize=(10, 10))
            for i in range(9):
                plt.subplot(330 + 1 + i)
                plt.suptitle("CutMix Images", fontsize=14)
                # plt.title(labels[targets[i]], fontsize=10)
                plt.imshow(data[i].permute(1, 2, 0))
            plt.show()
            """
            """
            print("Images with Cutout: ")
            fig = plt.figure(figsize=(10, 10))
            for i in range(9):
                plt.subplot(330 + 1 + i)
                plt.suptitle("Image with cutout", fontsize=14)
                plt.title(labels[targets[i]], fontsize=10)
                plt.imshow(mask(data[i].permute(1, 2, 0)))
            plt.show()"""

        train_avg_acc = sum_acc / len(train_loader)

        # Saving model
        if args.save_model == 'True':
            if epoch % 10 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

        # Evaluate Network
        model.eval()
        sum_acc = 0
        for data, targets in val_loader:
            # for batch_idx, (data, targets) in enumerate(tqdm(val_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            val_acc, val_loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=False)
            sum_acc += val_acc
        val_avg_acc = sum_acc / len(val_loader)

        print(
            f"Epoch: {epoch + 1} \tTraining accuracy: {train_avg_acc:.2f} \n\t\tValidation accuracy: {val_avg_acc:.2f}")

        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Train Accuracy": train_avg_acc, "Validation Accuracy": val_avg_acc}, step=train_steps)

    train_preds = get_all_preds(model, loader=prediction_loader, device=device)
    print(f"Train predictions shape: {train_preds.shape}")
    print(f"The label the network predicts strongly: {train_preds.argmax(dim=1)}")
    predictions = train_preds.argmax(dim=1)

    # Most Confident Incorrect Predictions
    images, labels, probs = get_predictions(model, prediction_loader, device)

    pred_labels = torch.argmax(probs, 1)
    print(f"pred_labels: {pred_labels}")

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))
    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)

    n_images = 48
    if args.dataset == 'mixed' or args.dataset == 'bw' or args.dataset == 'sobel' or args.dataset == 'edge':
        classes = ['Agriculture', 'Barren_Land', 'Brick_Kilns', 'Forest_Orchard', 'Industry', 'Roads', 'Urban',
                   'Urban_Green_Space', 'Water']
    elif args.dataset == 'resisc45':
        # classes = os.lisdir(resisc45)
        classes = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral',
                   'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest',
                   'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
                   'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass',
                   'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout',
                   'runway', 'sea_ice', 'ship', 'snow_berg', 'sparse_residential', 'stadium', 'storage_tank',
                   'tennis_court', 'terrace', 'thermal_power_station', 'wetland']
    elif args.dataset == 'eurosat':
        # classes = os.listdir(eurosat)
        classes = ['Annual_Crop', 'Forest', 'Herbaceous_Vegetation', 'Highway', 'Industrial', 'Pasture',
                   'Permanent_Crop', 'Residential', 'River', 'Sea_Lake']
    elif args.dataset == 'realproduct' or args.dataset == 'productreal':
        classes = os.listdir("Product")
    elif args.dataset == 'realart' or args.dataset == 'artreal':
        classes = os.listdir("Art")
    elif args.dataset == 'realclipart' or args.dataset == 'clipartreal':
        classes = os.listdir('Clipart')
    elif args.dataset == 'productclipart' or args.dataset == 'clipartproduct':
        classes = os.listdir('Clipart')
    elif args.dataset == 'productart' or args.dataset == 'artproduct':
        classes = os.listdir("Art")
    elif args.dataset == 'artclipart' or args.dataset == 'clipartart':
        classes = os.listdir("Clipart")
    else:
        classes = ['Agriculture', 'Barren_Land', 'Brick_Kilns', 'Forest_Orchard', 'Industry', 'Urban',
                   'Urban_Green_Space', 'Water']

    plot_most_incorrect(incorrect_examples, classes, n_images)
    wandb.save('Most_Conf_Incorrect_Pred.png')

    # Principle Components Analysis (PCA)
    outputs, intermediates, labels = get_representations(model, train_loader, device)

    output_pca_data = get_pca(outputs)
    plot_representations(output_pca_data, labels, classes, "PCA")
    wandb.save('PCA.png')

    # Intermediate Principle Components Analysis (PCA)
    intermediate_pca_data = get_pca(intermediates)
    plot_representations(intermediate_pca_data, labels, classes, "INTPCA")
    wandb.save('Intermediate_PCA.png')

    # t-Distributed Stochastic Neighbor Embedding (t-SNE)
    n_images = 10_000

    output_tsne_data = get_tsne(outputs, n_images=n_images)
    plot_representations(output_tsne_data, labels, classes, "TSNE", n_images=n_images)
    wandb.save('TSNE.png')

    # Intermediate t-Distributed Stochastic Neighbor Embedding (t-SNE)
    intermediate_tsne_data = get_tsne(intermediates, n_images=n_images)
    plot_representations(intermediate_tsne_data, labels, classes, "INTTSNE", n_images=n_images)
    wandb.save('Intermediate_TSNE.png')

    plot_confusion_matrix(y_test, train_preds.argmax(dim=1), classes)
    wandb.save('Confusion_Matrix.png')

    # Confusion Matrix
    wandb.sklearn.plot_confusion_matrix(y_test, train_preds.argmax(dim=1), labels)
    # Class proportions
    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    precision, recall, f1_score, support = score(y_test, train_preds.argmax(dim=1))
    test_acc = accuracy_score(y_test, train_preds.argmax(dim=1))
    wandb.log({"Test Accuracy": test_acc})

    print(f"Test Accuracy: {test_acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1_score: {f1_score}")
    print(f"support: {support}")

    # Test data saved in Excel document
    df = DataFrame({'Test Accuracy': test_acc, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                    'support': support})
    df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
    df.to_csv('test.csv', index=False)
    compression_opts = dict(method='zip', archive_name='out.csv')
    df.to_csv('out.zip', index=False, compression=compression_opts)

    wandb.save('test.csv')
    wandb.save('my_checkpoint.pth.tar')
    wandb.save('Predictions.csv')


if __name__ == "__main__":
    main()
