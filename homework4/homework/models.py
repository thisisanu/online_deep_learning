from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


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

        # Concatenate → (B, 20, 2)
        x = torch.cat([track_left, track_right], dim=1)

        # Normalize per-sample
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        x = (x - mean) / std

        # Flatten → (B, 40)
        x = x.view(B, -1)

        out = self.net(x)

        # Output → (B, n_waypoints, 2)
        return out.view(B, self.n_waypoints, 2)


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

        # ------------------------------------------------------------------
        # Encode track boundary points (x,y) into d_model
        # Input tokens: 20 track points (10 left + 10 right)
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(2, d_model)

        # Positional embedding for 20 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, n_track * 2, d_model))

        # ------------------------------------------------------------------
        # Transformer encoder to understand road geometry
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ------------------------------------------------------------------
        # Learnable queries (one per future waypoint)
        # ------------------------------------------------------------------
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Cross-attention for decoding
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Final head → (x, y)
        self.out = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right, **kwargs):
        """
        Args:
            track_left  : (B, n_track, 2)
            track_right : (B, n_track, 2)

        Returns:
            (B, n_waypoints, 2)
        """
        B = track_left.size(0)

        # ---------------------------------------------------------------
        # 1) Prepare encoder input tokens
        # ---------------------------------------------------------------
        x = torch.cat([track_left, track_right], dim=1)      # (B, 20, 2)
        x = self.input_proj(x)                               # (B, 20, d_model)
        x = x + self.pos_embed                               # add positional encodings

        # ---------------------------------------------------------------
        # 2) Encode road geometry
        # ---------------------------------------------------------------
        memory = self.encoder(x)                             # (B, 20, d_model)

        # ---------------------------------------------------------------
        # 3) Prepare queries for waypoints
        # ---------------------------------------------------------------
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, n_waypoints, d_model)

        # ---------------------------------------------------------------
        # 4) Cross-attention: queries attend to encoded track
        # ---------------------------------------------------------------
        decoded = self.decoder(queries, memory)              # (B, n_waypoints, d_model)

        # ---------------------------------------------------------------
        # 5) Predict (x, y)
        # ---------------------------------------------------------------
        out = self.out(decoded)                              # (B, n_waypoints, 2)

        return out



class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
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
    model_name = None
