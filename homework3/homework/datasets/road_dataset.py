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
                    road_transforms.TrackProcessor(self.track),
                    # Augmentations applied consistently to image, depth, and mask
                    road_transforms.RandomHorizontalFlip(p=0.5),
                    road_transforms.RandomBrightnessContrast(p=0.3),
                    road_transforms.RandomRotate(limit=15, p=0.3),
                ]
            )
        else:
            raise ValueError(f"Invalid transform '{transform_pipeline}' specified!")

        return xform

    def __len__(self):
        return len(self.frames["location"])

    def __getitem__(self, idx):
    sample = self.data[idx]  # however you load your dict

    # Make sure arrays are contiguous
    for key in ['image', 'track', 'depth']:
        val = sample[key]
        if isinstance(val, np.ndarray):
            sample[key] = val.copy()  # contiguous copy

    return sample



def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    """
    Constructs the dataset/dataloader.
    Args:
        dataset_path (str): root folder containing episodes
        transform_pipeline (str): 'default' or 'aug'
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers
        batch_size (int): batch size
        shuffle (bool): shuffle dataset for training
    Returns:
        DataLoader or Dataset
    """
    dataset_path = Path(dataset_path)
    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]

    if not scenes and dataset_path.is_dir():
        scenes = [dataset_path]

    datasets = []
    for episode_path in sorted(scenes):
        datasets.append(RoadDataset(episode_path, transform_pipeline=transform_pipeline))
    dataset = ConcatDataset(datasets)

    print(f"Loaded {len(dataset)} samples from {len(datasets)} episodes")

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )
