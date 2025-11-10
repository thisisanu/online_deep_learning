import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class SuperTuxClassificationDataset(Dataset):
    """
    SuperTuxKart Classification Dataset.
    Uses labels.csv for loading image paths and class labels.
    Auto-generates labels.csv if missing based on folder structure.
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to classification_data/
            split (str): 'train' or 'val'
            transform (callable): torchvision transforms to apply
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.labels_file = os.path.join(self.root_dir, "labels.csv")

        # Generate labels.csv if it doesn't exist
        if not os.path.exists(self.labels_file):
            self._generate_labels_csv()

        # Load samples from labels.csv
        self.samples = self._load_labels()

    def _generate_labels_csv(self):
        """Automatically generate labels.csv using folder structure"""
        print(f"[INFO] labels.csv not found. Generating in {self.root_dir} ...")
        class_folders = sorted([f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        with open(self.labels_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for cls_name in class_folders:
                cls_folder = os.path.join(self.root_dir, cls_name)
                for img_file in os.listdir(cls_folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(cls_name, img_file)  # relative path
                        writer.writerow([img_path, self.class_to_idx[cls_name]])

    def _load_labels(self):
        """Load image paths and labels from labels.csv"""
        samples = []
        with open(self.labels_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(self.root_dir, row['filename'])
                label = int(row['label'])
                if os.path.exists(img_path):
                    samples.append((img_path, label))
                else:
                    print(f"[WARNING] Image not found: {img_path}")
        if len(samples) == 0:
            raise RuntimeError(f"No valid images found in {self.root_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transform(train=True):
    """Returns torchvision transforms for training or validation"""
    if train:
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transform

def load_data(root_dir, batch_size=32, train=True):
    """Returns a DataLoader for train/val set"""
    dataset = SuperTuxClassificationDataset(
        root_dir=root_dir,
        split='train' if train else 'val',
        transform=get_transform(train)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)

# Optional: for class names reference
def get_class_names(root_dir, split='train'):
    labels_file = os.path.join(root_dir, split, "labels.csv")
    print("[get_class_names] Looking for labels.csv at:", labels_file)

    class_names = set()
    with open(labels_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.add(row['label'])
    return sorted(list(class_names))
