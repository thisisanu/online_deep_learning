from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import road_transforms
from .road_utils import Track


class RoadDataset(Dataset):
    """
    SuperTux dataset for road detection
    """

    def __init__(
        self,
        episode_path: str,
        transform_pipeline: str = "default",
    ):
        super().__init__()

        self.episode_path = Path(episode_path)

        info = np.load(self.episode_path / "info.npz", allow_pickle=True)
        self.track = Track(**info["track"].item())
        self.frames: dict[str, np.ndarray] = {k: np.stack(v) for k, v in info["frames"].item().items()}
        self.transform = self.get_transform(transform_pipeline)

    def get_transform(self, transform_pipeline: str):
        """
        Returns a composed transform pipeline based on the string identifier.
        """
        if transform_pipeline == "default":
            xform = road_transforms.Compose(
                [
                    road_transforms.ImageLoader(self.episode_path),
                    road_transforms.DepthLoader(self.episode_path),
                    road_transforms.TrackProcessor(self.track),
                ]
            )
        elif transform_pipeline == "aug":
            # Augmentation pipeline for training
            xform = road_transforms.Compose(
                [
                    road_transforms.ImageLoader(self.episode_path),
                    road_transforms.DepthLoader(self.episode_path),
