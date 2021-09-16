"""
Author: Georgios Voulgaris
Date: 25/05/2021
Description: Test platform to validate the impact on model robustness of various data augmentation techniques on various
            Deep architectures using SSRP dataset. The dataset is comprised of aerial images from Ghaziabad India.
            Stratification method was used to split the data to train/validate: 80% (out of which train: 80% and
            validation: 20%), and test: 20% data.
            Architectures: 4, and 5 CNN, ResNet18, ResNet50, ResNet101, ResNet152, VGG11, VGG13, VGG16, VGG19.
            Images: compressed 50x50, 100x100, and 226x226 pixel images. Note that 50x50 was too small for the 5 CNNs.
            Test Procedure: 5 runs for each architecture for each of the compressed data. That is 5x50x50 for each
            architecture. Then the Interquartile range, using the median was plotted.
            Plots: Average GPU usage per architecture, Interquartile, and for each architecture an F1 Score heatmap
            for each class.
            Data augmentation techniques tested: Cutout, mixup, CutMix and AugMix
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
import argparse
from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from models.CNN import CNN4, CNN5
from models.VGG import VGG11, VGG13, VGG16, VGG19
from Utilities.Save import save_checkpoint, load_checkpoint
from Utilities.Data import DataRetrieve
from Utilities.config import train_transforms, val_transforms, test_transforms
from Utilities.Identity import Identity
from CutMix.Cutout import mask
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import wandb
import sys

# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=102)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--save-model", default=False)
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--augmentation", default="cutout", help="cutout, cutmix")
    parser.add_argument("--Augmentation", default="none", help="none, position, cutout")
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--requires-grad", default=False, help="freeze the parameters so that the gradients are not "
                                                               "computed in backward()")
    parser.add_argument("--architecture", default="cnn4", help="cnn4=CNN4, cnn5=CNN5, vgg11=VGG11, vgg13=VGG13,"
                                                                    "vgg16=VGG16, tlvgg16=pretrain VGG16, vgg19=VGG19,"
                                                                    "resnet18=ResNet18, tlresnet18= pretrain ResNet18,"
                                                                    "resnet50=ResNet50, resnet101=ResNet101,"
                                                                    "resnet152=ResNet=152, tlalexnet= pretrain AlexNet")

    return parser.parse_args()


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
    wandb.init(project="Augmentations", config=args)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Load Data
    dataset = ImageFolder("Training_Data_2018_2014")
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

    # Networks
    if args.architecture == 'cnn4':
        model = CNN4(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'cnn5':
        model = CNN5(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg11':
        model = VGG11(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg13':
        model = VGG13(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg16':
        model = VGG16(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'tlvgg16':
        # Load pretrain model and modify it
        model = torchvision.models.vgg16(pretrained=args.pretrained)
        print(f"Pretrained is set: {args.pretrained}")
        # Do not change the layers upto that point
        # Freeze the parameters so that the gradients are not computed in backward()
        for param in model.parameters():
            param.requires_grad = args.requires_grad
        print(f"requires_grad={args.requires_grad}")
        model.avgpool = Identity()
        # This will only train the last layers
        model.classifier = nn.Sequential(nn.Linear(32768, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 10))
        model.to(device)
    elif args.architecture == 'vgg19':
        model = VGG19(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'tlalexnet':
        model = torchvision.models.alexnet(pretrained=args.pretrained)
        print(f"Pretrained={args.pretrained}")
        for param in model.parameters():
            param.requires_grad = args.requires_grad
        print(f"requires_grad={args.requires_grad}")
        model.classifier = nn.Sequential(nn.Dropout(),
                                         nn.Linear(9216, 4096),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(4096, 4096),
                                         nn.ReLU(),
                                         nn.Linear(4096, 10))
        model.to(device)
    elif args.architecture == 'resnet18':
        model = ResNet18(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'tlresnet18':
        model = torchvision.models.resnet18(pretrained=args.pretrained)
        print(f"Pretrained={args.pretrained}")
        for param in model.parameters():
            param.requires_grad = args.requires_grad
        print(f"requires_grad={args.requires_grad}")
        model.classifier = nn.Linear(512*4, 10)
        model.to(device)
    elif args.architecture == 'resnet50':
        model = ResNet50(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'resnet101':
        model = ResNet101(in_channels=args.in_channels, num_classes=num_classes).to(device)
    else:
        model = ResNet152(in_channels=args.in_channels, num_classes=num_classes).to(device)

    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load model
    if args.load_model is True:
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
        if args.save_model is True:
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

    # Confusion Matrix
    wandb.sklearn.plot_confusion_matrix(y_test, train_preds.argmax(dim=1), labels)
    # Class proportions
    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    precision, recall, f1_score, support = score(y_test, train_preds.argmax(dim=1))
    test_acc = accuracy_score(y_test, train_preds.argmax(dim=1))
    wandb.log({"Test Accuracy": test_acc}, step=train_steps)

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
