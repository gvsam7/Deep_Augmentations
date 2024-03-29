import torchvision
from torch import nn

import models.ACDilGabCNN
# from models.ResNet import ResNet18, ResNet50, ResNet101, ResNet152
from models.ResNet import ResNet50, ResNet101, ResNet152
from models.ResNet18 import ResNet18
from models.DilResNet18 import DilResNet18
from models.GabResNet18 import GabResNet18
from models.DilGabResNet18 import DilGabResNet18
from models.MPDilGabResNet18 import MPDilGabResNet18
from models.CNN import CNN4, CNN5, OldCNN3, OldCNN4, OldCNN5, CNN2, CAMCNN2, SPPCNN
from models.AlexNet import AlexNet
from models.VGG import VGG11, VGG13, VGG16, VGG19
from models.BagNet import BagnetCustom32, BagnetCustom96Thin
from Utilities.Identity import Identity
from models.GaborCNN import GaborCNN
from models.GaborMPCNN import GaborCNNMaxP
from models.GaborCNNMixP import GaborCNNMixP
from models.CNN5MixP import CNN5MixP
from models.CNN5MaxP import CNN5MaxP
from models.DilGabCNN import DilGaborCNN
from models.ACDilGabCNN import ACDilGaborCNN


def networks(architecture, in_channels, num_classes, pretrained, requires_grad, global_pooling):
    if architecture == 'cnn4':
        model = CNN4(in_channels, num_classes)
    elif architecture == 'oldcnn3':
        model = OldCNN3(in_channels, num_classes)
    elif architecture == 'oldcnn4':
        model = OldCNN4(in_channels, num_classes)
    elif architecture == 'cnn2':
        model = CNN2(in_channels, num_classes)
    elif architecture == 'oldcnn5':
        model = OldCNN5(in_channels, num_classes)
    elif architecture == 'sppcnn':
        model = SPPCNN(in_channels, num_classes)
    elif architecture == 'gaborcnn':
        model = GaborCNN(in_channels, num_classes)
    elif architecture == 'gaborcnnmaxp':
        model = GaborCNNMaxP(in_channels, num_classes)
    elif architecture == 'gaborcnnmixp':
        model = GaborCNNMixP(in_channels, num_classes)
    elif architecture == 'dilgabcnn':
        model = DilGaborCNN(in_channels, num_classes)
    elif architecture == 'acdilgabcnn':
        model = ACDilGaborCNN(in_channels, num_classes)
    elif architecture == 'cnn5mixp':
        model = CNN5MixP(in_channels, num_classes)
    elif architecture == 'cnn5maxp':
        model = CNN5MaxP(in_channels, num_classes)
    elif architecture == 'camcnn2':
        model = CAMCNN2(in_channels, num_classes)
    elif architecture == 'alexnet':
        model = AlexNet(in_channels, num_classes)
    elif architecture == 'cnn5':
        model = CNN5(in_channels, num_classes)
    elif architecture == 'vgg11':
        model = VGG11(in_channels, num_classes)
    elif architecture == 'vgg13':
        model = VGG13(in_channels, num_classes)
    elif architecture == 'vgg16':
        model = VGG16(in_channels, num_classes)
    elif architecture == 'tlvgg16':
        # Load pretrain model and modify it
        model = torchvision.models.vgg16(pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained={pretrained}")
            # Do not change the layers upto that point
            # Freeze the parameters so that the gradients are not computed in backward()
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad={requires_grad}")
            if global_pooling == "GP":
                print(f"Global Pooling: {global_pooling}")
                model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 10))
            else:
                print(f"Global Pooling: {global_pooling}")
                model.avgpool = Identity()
                # This will only train the last layers
                model.classifier = nn.Sequential(nn.Linear(32768, 100),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(100, 10))
        else:
            print(f"Fully trained from SSRP data, Pretrained={pretrained}")
    elif architecture == 'vgg19':
        model = VGG19(in_channels, num_classes)
    elif architecture == 'BagnetCustom32':
        model = BagnetCustom32(in_channels, num_classes, input_size=32)
    elif architecture == 'BagnetCustom96Thin':
        model = BagnetCustom96Thin(in_channels, num_classes, input_size=96)
    elif architecture == 'tlalexnet':
        model = torchvision.models.alexnet(pretrained=pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained={pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad={requires_grad}")
            if global_pooling == "GP":
                print(f"Global Pooling: {global_pooling}")
                model.classifier = nn.Sequential(nn.Dropout(),
                                                 nn.Linear(9216, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU(),
                                                 nn.Linear(4096, 10))
            else:
                print(f"Global Pooling: {global_pooling}")
                model.avgpool = Identity()
                model.classifier = nn.Sequential(nn.Dropout(),
                                                 nn.Linear(12544, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU(),
                                                 nn.Linear(4096, 10))
        else:
            print(f"Fully trained from SSRP data, Pretrained={pretrained}")
    elif architecture == 'resnet18':
        model = ResNet18(in_channels, num_classes)
    elif architecture == 'dilresnet18':
        model = DilResNet18(in_channels, num_classes)
    elif architecture == 'gabresnet18':
        model = GabResNet18(in_channels, num_classes)
    elif architecture == 'dilgabresnet18':
        model = DilGabResNet18(in_channels, num_classes)
    elif architecture == 'mpdilgabresnet18':
        model = MPDilGabResNet18(in_channels, num_classes)
    elif architecture == 'tlresnet18':
        model = torchvision.models.resnet18(pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained={pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad={requires_grad}")
            model.classifier = nn.Linear(512*4, 10)
        else:
            print(f"Fully trained from SSRP data, Pretrained={pretrained}")
    elif architecture == 'resnet50':
        model = ResNet50(in_channels, num_classes)
    elif architecture == 'resnet101':
        model = ResNet101(in_channels, num_classes)
    else:
        model = ResNet152(in_channels, num_classes)
    return model
