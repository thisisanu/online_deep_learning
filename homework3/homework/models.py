# ───────────────────────────────
# Fixed Classifier
# ───────────────────────────────
class Classifier(nn.Module):
    """
    Convolutional image classifier for the SuperTux dataset.
    Works with arbitrary input sizes using AdaptiveAvgPool2d.
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

        # Adaptive pooling ensures fixed feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Convolutional + residual layers
        x = F.relu(self.bn0(self.conv0(z)))
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.pool3(self.res3(x))

        # Adaptive pooling + flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict(self, x):
        return self(x).argmax(dim=1)


# ───────────────────────────────
# Fixed Detector
# ───────────────────────────────
class Detector(nn.Module):
    """
    Detector for joint segmentation (3 classes) and depth estimation.
    Output is upsampled to match input size exactly.
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
        # Normalize input
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

        # Ensure outputs match input size exactly
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        depth = F.interpolate(depth.unsqueeze(1), size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        return logits, depth

    def predict(self, x):
        logits, depth = self(x)
        return logits.argmax(dim=1), depth


# ───────────────────────────────
# Debug
# ───────────────────────────────
def debug_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = torch.rand(1, 3, 96, 128).to(device)
    model_cls = Classifier().to(device)
    model_det = Detector().to(device)

    model_cls.eval()
    model_det.eval()

    logits_cls = model_cls(sample)
    logits_det, depth_det = model_det(sample)

    print("Classifier logits:", logits_cls.shape)
    print("Detector logits:", logits_det.shape, "Depth:", depth_det.shape)


if __name__ == "__main__":
    debug_model()
