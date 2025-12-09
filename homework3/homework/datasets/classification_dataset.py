import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    """
    SuperTux dataset for classification
    """

    def __init__(
        self,
        dataset_path: str,
        transform_pipeline: str = "default",
    ):
        self.transform = self.get_transform(transform_pipeline)
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    img_path = Path(dataset_path, fname)
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((img_path, label_id))

    def get_transform(self, transform_pipeline: str = "default"):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.ToTensor()
        elif transform_pipeline == "aug":
            # construct your custom augmentation
            xform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                ]
            )

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Pairs of images and labels (int) for classification
        """
        img_path, label_id = self.data[idx]
        img = Image.open(img_path)
        data = (self.transform(img), label_id)

        return data


from pathlib import Path
from torch.utils.data import DataLoader

def load_data(
    dataset_path: str | Path,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 128,
) -> tuple:
    """
    Constructs train and val datasets/dataloaders.
    The specified transform_pipeline must be implemented in the SuperTuxDataset class.

    Args:
        dataset_path (str | Path): root folder containing 'train' and 'val' subfolders
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for debugging
        batch_size (int): batch size

    Returns:
        Tuple of (train_dataset_or_loader, val_dataset_or_loader)
    """
    dataset_path = Path(dataset_path)
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected folders 'train' and 'val' inside {dataset_path}"
        )

    train_dataset = SuperTuxDataset(train_path, transform_pipeline=transform_pipeline)
    val_dataset = SuperTuxDataset(val_path, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return train_dataset, val_dataset

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader

