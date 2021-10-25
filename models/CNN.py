from torch import nn
import torch.nn.functional as F
# from Pool.MedianPool import MedianPool2d


class CNN4(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # self.pool3 = MedianPool2d(kernel_size=2, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.pool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, num_classes)
        # self.fc1 = nn.Linear(256 * 16 * 16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        # x = self.pool4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        # x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class OldCNN3(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(OldCNN3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.fc1 = nn.Linear(128 * 32 * 32, 512)
        # self.fc1 = nn.Linear(18432, 512)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.flatten(1)
        # x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OldCNN4(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(OldCNN4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            # nn.Linear(256 * 3 * 3, 512),  # 50x50 (1,573,193 trainable parameters)
            # nn.Linear(256 * 6 * 6, 512),  # 100x100 (5,112,137 trainable parameters)
            nn.Linear(256 * 16 * 16, 512),  # 256x256 (33,947,977 trainable parameters)
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class OldCNN5(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(OldCNN5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),  # 100x100 (3,933,001 trainable parameters)
            # nn.Linear(512 * 8 * 8, 512),  # 256x256 (18,350,921 trainable parameters)
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0, 1),
            nn.Conv2d(64, 192, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0, 1),
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),  # 100x100 ( 23,486,281 trainable parameters)
            # nn.Linear(256 * 7 * 7, 4096),  # 256x256 (70,672,201  trainable parameters)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    """
    # For plotting intermediate PCA and t-SNE
    def forward(self, x):
        x = self.features(x)
        h = x.flatten(1)
        x = self.classifier(h)
        return x, h"""


class FCNN4(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(FCNN4, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)


class CNN5(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Batch Normalisation
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # self.pool5 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * 8 * 8, num_classes)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        # x = self.pool5(x)

        x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        x = x.flatten(1)
        x = self.fc1(x)
        return x
