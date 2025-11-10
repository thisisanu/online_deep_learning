from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# ──────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────
class Classifier(nn.Module):
    """
    Convolutional image classifier for the SuperTux dataset.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )

        self.feature_size = 128 * (64 // 8) * (64 // 8)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        feats = self.conv_layers(z)
        feats = torch.flatten(feats, 1)
        return self.classifier(feats)

    def predict(self, x):
        return self(x).argmax(dim=1)


# ──────────────────────────────────────────────────────────────
# Detector (Segmentation + Depth)
# ──────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Helper block: Conv + BN + ReLU"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Detector(nn.Module):
    """
    Detector for joint segmentation (3 classes) and depth estimation.
    Input: (B, 3, H, W)
    Outputs:
      - logits: (B, 3, H, W)
      - depth:  (B, H, W)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder
        self.down1 = ConvBlock(in_channels, 16, stride=2)
        self.down2 = ConvBlock(16, 32, stride=2)
        self.down3 = ConvBlock(32, 64, stride=2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(16, 16, 2, stride=2)

        # Output heads
        self.seg_head = nn.Conv2d(16, num_classes, 1)
        self.depth_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        x1 = self.down1(z)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = F.relu(self.up1(x3))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))

        logits = self.seg_head(x)
        depth = self.depth_head(x).squeeze(1)  # (B, H, W)
        return logits, depth

    def predict(self, x):
        logits, depth = self(x)
        return logits.argmax(dim=1), depth


# ──────────────────────────────────────────────────────────────
# Model utilities
# ──────────────────────────────────────────────────────────────
MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    """Loads model by name and optionally its weights."""
    m = MODEL_FACTORY[model_name](**model_kwargs)
    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
    return m


def save_model(model: nn.Module) -> str:
    """Save model weights to disk (classifier.th or detector.th)."""
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if isinstance(model, m):
            model_name = n
    if model_name is None:
        raise ValueError(f"Unsupported model type: {type(model)}")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Saved model to {output_path}")
    return str(output_path)


def calculate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.rand(1, 3, 96, 128).to(device)
    model = Detector().to(device)
    model.eval()
    logits, depth = model(sample)
    print("Logits:", logits.shape, "Depth:", depth.shape)


if __name__ == "__main__":
    debug_model()
