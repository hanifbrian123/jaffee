"""JAFFE models. 2D analogs of the CASME II video backbones.

The CASME II winner was r3d_18 (3D CNN, Kinetics-pretrained). Static images have
no temporal axis, so the transferable analog is a strong ImageNet-pretrained 2D
CNN. A tiny from-scratch CNN is kept as the Kaggle-style control.
"""
import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    """Small from-scratch CNN — the Kaggle-notebook-style control."""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(cfg, num_classes):
    name = cfg["backbone"]
    dropout = cfg.get("dropout", 0.5)
    pretrained = cfg.get("pretrained", True)
    if name == "simplecnn":
        return SimpleCNN(num_classes, dropout=dropout)
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Sequential(nn.Dropout(dropout),
                               nn.Linear(net.fc.in_features, num_classes))
        return net
    if name == "resnet34":
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        net.fc = nn.Sequential(nn.Dropout(dropout),
                               nn.Linear(net.fc.in_features, num_classes))
        return net
    raise ValueError(f"backbone tak dikenal: {name}")
