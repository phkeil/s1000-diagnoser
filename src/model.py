"""Model architectures, extracted verbatim from the notebooks that trained them
so training and serving code share one definition instead of each redefining it.

MotorAutoencoder: notebooks/04_CAE_Annomalydetector.ipynb (the production anomaly
detector — trained on healthy-only mel spectrograms, scored by reconstruction MSE).

EfficientDiagnoser: notebooks/07_EfficiencyNet.ipynb (secondary supervised
healthy/defective classifier, not currently wired into the anomaly pipeline).
"""

import torch
import torch.nn as nn
from torchvision import models


class MotorAutoencoder(nn.Module):
    """Convolutional autoencoder over 224x224x3 mel-spectrogram images.

    Encoder downsamples 224 -> 14 (stride-2 convs) into a `bottleneck_size`
    vector; decoder mirrors it back to 224x224 via transposed convs. Trained
    with MSE reconstruction loss on healthy-only data; anomaly score is the
    reconstruction MSE at inference time (see src/inference.py).
    """

    def __init__(self, bottleneck_size: int = 128):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.flatten = nn.Flatten()
        self.fc_bottleneck = nn.Linear(512 * 14 * 14, bottleneck_size)
        self.bottleneck_bn = nn.BatchNorm1d(bottleneck_size)
        self.relu = nn.ReLU()

        self.decoder_fc = nn.Linear(bottleneck_size, 512 * 14 * 14)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),  # 224x224
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.fc_bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)

        x = self.decoder_fc(x)
        x = x.view(-1, 512, 14, 14)
        x = self.decoder_conv(x)
        return x


class EfficientDiagnoser(nn.Module):
    """EfficientNet-B0-backed binary (healthy/defective) classifier.

    Returns (logits, features) — features are the pooled EfficientNet
    embedding before the custom head, kept for downstream analysis
    (e.g. embedding-distance approaches) as in the source notebook.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.base = models.efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.base(x)
        logits = self.head(features)
        return logits, features
