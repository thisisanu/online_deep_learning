from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from . import road_transforms
from .road_utils import Track


class RoadDataset(Dataset):
    """
    SuperTux dataset for road detection
    """

    def __init__(self, episode_path: str, transform_pipeline: str = "default"):
        super().__init__()
        self.episode_path = Path(episode_path)

        # Load episode info
        info = np.load(self.episode_path / "info.npz", allow_pickle=True)
        self.track = Track(**info["track"].item())
        self.frames: dict[str, np.ndarray] = {k: np.stack(v) for k, v in info["frames"].item()}

        # Setup transform pipeline
        self.transform = self.get_transform(transform_pipeline)

    def get_transform(self, transform_pipeline: str):
        """
        Returns a composed transform pipeline based on the string identifier.
        """
        if transform_pipeline == "default":
            xform = road_transforms.Compose([
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
                road_transforms.TrackProcessor(self.track),
            ])
        elif transform_pipeline == "aug":
            xform = road_transforms.Compose([
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
                road_transforms.TrackProcessor(self.track),
                road_transforms.RandomHorizontalFlip(p=0.5),
                road_transforms.RandomBrightnessContrast(p=0.3),
                road_transforms.RandomRotate(limit=15, p=0.3),
            ])
        else:
            raise ValueError(f"Invalid transform '{transform_pipeline}' specified!")

        return xform

    def __len__(self):
        return len(self.frames["location"])

    def __getitem__(self, idx):
        # Create sample dictionary from frames
        sample = {k: self.frames[k][idx] for k in self.frames}
        sample['_idx'] = idx  # required for ImageLoader and other transforms

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        # Ensure all arrays are contiguous to prevent stride issues
        for key in ['image', 'track', 'depth']:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = sample[key].copy()

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
    Constructs the dataset or dataloader.
    """
    dataset_path = Path(dataset_path)
    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]
    if not scenes and dataset_path.is_dir():
        scenes = [dataset_path]

    datasets = [RoadDataset(ep, transform_pipeline=transform_pipeline) for ep in sorted(scenes)]
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
