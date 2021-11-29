from torch import nn


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes=9):
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
            # nn.Linear(256 * 2 * 2, 4096),  # 100x100 ( 23,486,281 trainable parameters)
            nn.Linear(256 * 7 * 7, 4096),  # 256x256 (70,672,201  trainable parameters)
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
