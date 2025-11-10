
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import argparse
import torchvision.models as models

# Device configuration
def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

# Data transformations
def get_data_transforms(image_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

# Data loading
def get_data_loaders(data_dir, data_transforms, batch_size):
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Found {dataset_sizes['train']} training images and {dataset_sizes['val']} validation images.")
    print(f"Classes: {class_names}")
    return dataloaders, dataset_sizes, class_names

# Define the Classification Model
def create_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Define Loss Function and Optimizer
def setup_training_components(model, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification Training Setup')
    parser.add_argument('--data_dir', type=str, default='./classification_data', help='Directory with classification data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for resizing')
    args = parser.parse_args()

    device = setup_device()
    data_transforms = get_data_transforms(args.image_size)
    dataloaders, dataset_sizes, class_names = get_data_loaders(args.data_dir, data_transforms, args.batch_size)

    # Model, Loss, and Optimizer setup
    num_classes = len(class_names)
    model = create_model(num_classes).to(device)
    criterion, optimizer = setup_training_components(model)

    print(f"Model: {model.__class__.__name__} with {num_classes} output classes")
    print(f"Loss Function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")

    # Example of how to access data
    # inputs, labels = next(iter(dataloaders['train']))
    # print(f"Batch input shape: {inputs.shape}")
    # print(f"Batch labels shape: {labels.shape}")

    print("Classification training setup complete.")
