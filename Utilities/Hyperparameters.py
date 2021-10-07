import argparse


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
    parser.add_argument("--requires-grad", default=False)
    parser.add_argument("--global-pooling", default=None)
    parser.add_argument("--architecture", default="cnn4", help="cnn4=CNN4, cnn5=CNN5, vgg11=VGG11, vgg13=VGG13, oldcnn3=OldCNN3"
                                                                    "vgg16=VGG16, tlvgg16=pretrain VGG16, vgg19=VGG19,"
                                                                    "resnet18=ResNet18, tlresnet18= pretrain ResNet18,"
                                                                    "resnet50=ResNet50, resnet101=ResNet101,"
                                                                    "resnet152=ResNet=152, tlalexnet= pretrain AlexNet")

    return parser.parse_args()