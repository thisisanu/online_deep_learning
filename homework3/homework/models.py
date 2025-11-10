from pathlib import Path
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    """
    Convolutional image classifier for the SuperTux dataset.

    Input:
        (B, 3, 64, 64)
    Output:
        (B, 6) logits

    ⚠️ Remember:
        - Call `model.train()` before training
        - Call `model.eval()` before validation/testing
          (even when using `torch.inference_mode()`).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        # Flattened feature size for 64×64 input → (8×8 after 3× MaxPool2d(2))
        self.feature_size = 128 * (64 // 8) * (64 // 8)

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        Args:
            x (B, 3, H, W): input image tensor in [0, 1]
        Returns:
            logits (B, num_classes)
        """
        # Normalize
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Feature extraction
        features = self.conv_layers(z)

        # Flatten and classify
        features = torch.flatten(features, 1)
        logits = self.classifier(features)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: return class labels (argmax of logits)
        """
        return self(x).argmax(dim=1)


# ──────────────────────────────────────────────────────────────
# Detector placeholder for future assignment
# ──────────────────────────────────────────────────────────────
class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        # TODO (for later): implement actual detection model
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))
        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        depth = raw_depth
        return pred, depth


# ──────────────────────────────────────────────────────────────
# Model loading utilities
# ──────────────────────────────────────────────────────────────
MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    """
    Loads a model by name.
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}. Check model arguments."
            ) from e

    # Limit model size to 20 MB (for submission)
    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: nn.Module) -> str:
    """
    Saves model weights to disk.
    """
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if isinstance(model, m):
            model_name = n
    if model_name is None:
        raise ValueError(f"Unsupported model type {type(model)}")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: nn.Module) -> float:
    """Compute model size in megabytes."""
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Quick check for model input/output shapes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)
    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    model.eval()  # <- Always set to eval for testing/inference
    output = model(sample_batch)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
