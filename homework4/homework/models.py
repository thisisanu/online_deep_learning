from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(40, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.longitudinal_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        
        self.lateral_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        left_flat = track_left.view(track_left.shape[0], -1)
        right_flat = track_right.view(track_right.shape[0], -1)
        
        x = torch.cat([left_flat, right_flat], dim=1)
        
        features = self.feature_extractor(x)
        
        longitudinal_coords = self.longitudinal_head(features)
        lateral_coords = self.lateral_head(features)
        
        waypoints = torch.stack([longitudinal_coords, lateral_coords], dim=-1)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        input_size = 2 * n_track * 2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, d_model)
        )
        
        self.longitudinal_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        
        self.lateral_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = track_left.shape[0]
        original_device = track_left.device

        track_left = track_left.cpu()
        track_right = track_right.cpu()
        self.cpu()

        left_flat = track_left.view(batch_size, -1)
        right_flat = track_right.view(batch_size, -1)
        
        x = torch.cat([left_flat, right_flat], dim=1)
        
        features = self.feature_extractor(x)
        
        longitudinal_coords = self.longitudinal_head(features)
        lateral_coords = self.lateral_head(features)
        
        waypoints = torch.stack([longitudinal_coords, lateral_coords], dim=-1)
        
        if original_device != torch.device('cpu'):
            waypoints = waypoints.to(original_device)
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        features = self.features(x)
        
        waypoint_coords = self.regressor(features)
        
        waypoints = waypoint_coords.view(waypoint_coords.shape[0], self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "linear_planner": MLPPlanner,
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
        assert model_path.exists(), f"{model_path.as_posix()} not found"

        try:
            if model_name == "transformer_planner":
                state_dict = torch.load(model_path, map_location='cpu')
                m.load_state_dict(state_dict)
                m = m.to('cpu')
            else:
                state_dict = torch.load(model_path, map_location="cpu")
                m.load_state_dict(state_dict)
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.as_posix()}, make sure the file is not corrupted"
            ) from e

    return m


def save_model(model: torch.nn.Module) -> str:
    model_name = None
    
    name_priority = ["mlp_planner", "transformer_planner", "cnn_planner", "linear_planner"]

    for n in name_priority:
        if n in MODEL_FACTORY and type(model) is MODEL_FACTORY[n]:
            model_name = n
            break

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return str(output_path)


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
