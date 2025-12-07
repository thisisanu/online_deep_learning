from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# ---------------------------------------------------------------
# MLP Planner
# ---------------------------------------------------------------
class MLPPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2  # (left/right) * (x/y)
        output_dim = n_waypoints * 2
        hidden_dim = 320

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, track_left, track_right, **kwargs):
        B = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        x = (x - mean) / std
        x = x.view(B, -1)
        out = self.net(x)
        return out.view(B, self.n_waypoints, 2)


# ---------------------------------------------------------------
# Transformer Planner
# ---------------------------------------------------------------
class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Encode track points
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_track * 2, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.out = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right, **kwargs):
        B = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1)
        x = self.input_proj(x)
        x = x + self.pos_embed
        memory = self.encoder(x)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        decoded = self.decoder(queries, memory)
        out = self.out(decoded)
        return out


# ---------------------------------------------------------------
# CNN Planner (stub, not implemented)
# ---------------------------------------------------------------
class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        raise NotImplementedError


# ---------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------
MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


# ---------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------
def calculate_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / 1024**2


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)
    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: nn.Module) -> str:
    model_name = model.__class__.__name__.lower() + ".th"
    path = HOMEWORK_DIR / model_name
    torch.save(model.state_dict(), path)
    print(f"Saved model → {path}")
    return str(path)
