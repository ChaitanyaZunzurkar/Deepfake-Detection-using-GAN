"""Model definitions matching the user's custom-GAN notebook."""
import torch
from torch import nn
from torchvision import models as tv_models

class ResidualBlock(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return x + 0.2 * self.block(x)

class EnhancementGenerator(nn.Module):
    def __init__(self, channels=48, blocks=6):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, channels, 9, 1, 4), nn.PReLU(channels))
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.body_tail = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels))
        self.output = nn.Conv2d(channels, 3, 9, 1, 4)
    def forward(self, x):
        features = self.head(x)
        refined = features + self.body_tail(self.body(features))
        return (x + 0.25 * torch.tanh(self.output(refined))).clamp(0.0, 1.0)

def make_efficientnet():
    # Checkpoint loading overwrites all parameters; no pretrained download is needed.
    try:
        net = tv_models.efficientnet_b0(weights=None)
    except TypeError:
        net = tv_models.efficientnet_b0(pretrained=False)
    dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return net, dim

class SingleInputClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, dim = make_efficientnet()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 1))
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, dim = make_efficientnet()
        self.head = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(dim * 2, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(256, 1)
        )
    def forward(self, original, enhanced):
        f1 = self.backbone(original)
        f2 = self.backbone(enhanced)
        return self.head(torch.cat([f1, f2], dim=1)).squeeze(1)
