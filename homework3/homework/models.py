import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

# ───────────────────────────────
# Residual Block
# ───────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm2d(out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.bn_shortcut(self.shortcut(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

# ───────────────────────────────
# Classifier
# ───────────────────────────────
class Classifier(nn.Module):
    """
    Convolutional image classifier for the SuperTux dataset.
    Works with arbitrary input sizes.
    """
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

        # Initial conv layer
        self.conv0 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(32)

        # Residual blocks
        self.res1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Fully connected classifier (initialized dynamically)
        self.fc1 = None
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = F.relu(self.bn0(self.conv0(z)))
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.pool3(self.res3(x))
        x = torch.flatten(x, 1)

        # Initialize fc1 dynamically based on input feature size
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 512).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict(self, x):
        return self(x).argmax(dim=1).long()

# ───────────────────────────────
# Detector (Segmentation + Depth)
# ───────────────────────────────
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    """Upsample with skip connection"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Detector(nn.Module):
    """
    Detector for joint segmentation (3 classes) and depth estimation.
    Works with arbitrary input sizes.
    """
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

        # Encoder
        self.down1 = ConvBlock(in_channels, 16)
        self.down2 = ConvBlock(16, 32)
        self.down3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up1 = UpBlock(64, 32, 32)
        self.up2 = UpBlock(32, 16, 16)

        # Heads
        self.seg_head = nn.Conv2d(16, num_classes, 1)
        self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        d1 = self.down1(z)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        # Decoder
        u1 = self.up1(d3, d2)
        u2 = self.up2(u1, d1)

        # Heads
        logits = self.seg_head(u2)
        depth = torch.sigmoid(self.depth_head(u2)).squeeze(1)
        return logits, depth

    def predict(self, x):
        logits, depth = self(x)
        return logits.argmax(dim=1).long(), depth

# ───────────────────────────────
# Model Factory & Utilities
# ───────────────────────────────
MODEL_FACTORY = {"classifier": Classifier, "detector": Detector}

def load_model(model_name: str, with_weights=False, **kwargs):
    m = MODEL_FACTORY[model_name](**kwargs)
    if with_weights:
        path = HOMEWORK_DIR / f"{model_name}.th"
        assert path.exists(), f"{path} not found"
        m.load_state_dict(torch.load(path, map_location="cpu"))
    return m

def save_model(model: nn.Module) -> str:
    for name, cls in MODEL_FACTORY.items():
        if isinstance(model, cls):
            path = HOMEWORK_DIR / f"{name}.th"
            torch.save(model.state_dict(), path)
            print(f"[INFO] Saved model to {path}")
            return str(path)
    raise ValueError(f"Unsupported model type: {type(model)}")
