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
    parser.add_argument("--dataset", default="mixed", help="mixed, bw, edge, sobel, wet, dry, bw_wet, edge_wet, "
                                                           "sobel_wet, bw_dry, edge_dry, sobel_dry, resisc45, eurosat,"
                                                           "realproduct, productreal, realart, artreal, realclipart,"
                                                           "clipartreal, productclipart, clipartproduct, productart,"
                                                           "artproduct")
    parser.add_argument("--architecture", default="cnn4", help="cnn2=CNN2, cnn4=CNN4, cnn5=CNN5, vgg11=VGG11, vgg13=VGG13, oldcnn3=OldCNN3"
                                                                    "camcnn2=CAMCNN2, oldcnn4=OldCNN4, oldcnn5=OldCNN5, vgg16=VGG16,"
                                                                    "tlvgg16=pretrain VGG16, vgg19=VGG19, alexnet=AlexNet"
                                                                    "resnet18=ResNet18, tlresnet18=pretrain ResNet18,"
                                                                    "dilresnet18=DilResNet18, gabresnet18=GabResNet18,"
                                                                    "dilgabresnet18=DilGabResNet18,"
                                                                    "mpdilgabresnet18=MPDilFGabResNet18"
                                                                    "resnet50=ResNet50, resnet101=ResNet101,"
                                                                    "resnet152=ResNet=152, tlalexnet=pretrain AlexNet,"
                                                                    "sppcnn=SSPCNN, BagnetCustom32, BagnetCustom96Thin,"
                                                                    "gaborcnn=GaborCNN, gaborcnnmaxp=GaborCNNMaxP,"
                                                                    "cnn5mixp=CNN5MixP, cnn5maxp=CNN5MaxP,"
                                                                    "gaborcnnmixp=GaborCNNMixP, dilgabcnn = DilGaborCNN,"
                                                               "acdilgabcnn = ACDilGaborCNN")

    return parser.parse_args()