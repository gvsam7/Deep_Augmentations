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
import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize
import argparse
from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from models.CNN import CNN
from models.VGG import VGG11, VGG13, VGG16, VGG19
from Utilities.Save import save_checkpoint, load_checkpoint
from pandas import DataFrame
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import wandb
wandb.init(project="Augmentations")

# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--architecture", default="cnn", help="cnn=CNN, vgg11=VGG11, vgg13=VGG13, vgg16=VGG16,"
                                                                  "vgg19=VGG19, resnet18=ResNet18, resnet50=ResNet50,"
                                                              "resnet101=ResNet101, resnet152=ResNet=152")

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

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # transforms = transform_data
    transforms = Compose([Resize((args.height, args.width)), ToTensor()])
    # Load Data
    dataset = ImageFolder("Training_Data_2018_2014", transform=transforms)
    labels = dataset.classes
    num_classes = len(labels)
    print(f"num_class: {num_classes}")
    input_size = dataset[0][0].shape
    X1, y1 = dataset, dataset.targets
    X_trainval, X_test, y_trainval, y_test = train_test_split(X1, y1, test_size=0.2, stratify=y1,
                                                              random_state=args.random_state, shuffle=True)
    test = X_test
    X2 = X_trainval
    y2 = y_trainval
    X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2, stratify=y2,
                                                      random_state=args.random_state, shuffle=True)
    train, val = X_train, X_val
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    prediction_loader = DataLoader(test, batch_size=args.batch_size)

    # Initialize network
    if args.architecture == 'cnn':
        model = CNN(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg11':
        model = VGG11(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg13':
        model = VGG13(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg16':
        model = VGG16(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'vgg19':
        model = VGG19(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.architecture == 'resnet18':
        model = ResNet18(in_channels=args.in_channels, num_classes=num_classes).to(device)
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

    if args.load_model:
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
            acc, loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=True)
            sum_acc += acc
        train_avg_acc = sum_acc / len(train_loader)

        if epoch % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        model.eval()
        sum_acc = 0
        for data, targets in val_loader:
        # for batch_idx, (data, targets) in enumerate(tqdm(val_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            val_acc, val_loss = step(data, targets, model=model, optimizer=optimizer, criterion=criterion, train=True)
            sum_acc += val_acc
        val_avg_acc = sum_acc / len(val_loader)

        print(f"Epoch: {epoch + 1} \tTraining accuracy: {train_avg_acc:.2f} \n\t\tValidation accuracy: {val_avg_acc:.2f}")

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
