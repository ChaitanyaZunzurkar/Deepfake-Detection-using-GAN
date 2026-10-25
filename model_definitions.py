"""
Model definitions for the GAN-based deepfake detection pipeline.

Architectures reconstructed from:
    - deepfake-detection(6).ipynb
    - training-gan-for-enhancement(1).ipynb

Class mapping from the detection notebook:
    0 = Real
    1 = Fake
"""

import torch
from torch import nn
from torchvision import models


# ============================================================
# GAN: Enhancement Generator
# Source: training-gan-for-enhancement(1).ipynb, cells 14-15
# ============================================================

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

        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 9, 1, 4),
            nn.PReLU(channels),
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(blocks)
            ]
        )

        self.body_tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

        self.output = nn.Conv2d(
            channels,
            3,
            9,
            1,
            4,
        )

    def forward(self, x):
        features = self.head(x)

        refined = features + self.body_tail(
            self.body(features)
        )

        return (
            x + 0.25 * torch.tanh(self.output(refined))
        ).clamp(0.0, 1.0)


# ============================================================
# EfficientNet-B0 Feature Extractor
# Source: deepfake-detection(6).ipynb, cell 30
# ============================================================

def make_backbone(pretrained=False):
    """
    Create EfficientNet-B0 with its original classifier removed.

    pretrained=False prevents downloading ImageNet weights during
    inference. The trained checkpoint supplies the model parameters.
    """

    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()

    return backbone


# ============================================================
# M1: Original Image Classifier
# M2: GAN-Enhanced Image Classifier
# Source: deepfake-detection(6).ipynb, cell 30
# ============================================================

class SingleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = make_backbone(pretrained=False)
        self.head = nn.Linear(1280, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ============================================================
# M3: Original + Enhanced Feature Fusion Classifier
# Source: deepfake-detection(6).ipynb, cell 31
# ============================================================

class FusionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.original = make_backbone(pretrained=False)
        self.enhanced = make_backbone(pretrained=False)

        self.head = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, original, enhanced):
        original_features = self.original(original)
        enhanced_features = self.enhanced(enhanced)

        combined = torch.cat(
            [original_features, enhanced_features],
            dim=1,
        )

        return self.head(combined)


# ============================================================
# Model Factory
# ============================================================

def create_model(model_name):
    """
    Create a classifier using the architecture from the notebook.

    Supported names:
        M1_original
        M2_gan_enhanced
        M3_feature_fusion
    """

    if model_name in {
        "M1_original",
        "M2_gan_enhanced",
        "cnn_original",
        "cnn_enhanced",
        "original_cnn",
        "gan_enhanced_cnn",
    }:
        return SingleCNN()

    if model_name in {
        "M3_feature_fusion",
        "cnn_feature_fusion",
        "feature_fusion",
    }:
        return FusionCNN()

    raise ValueError(
        f"Unknown model name: {model_name}"
    )