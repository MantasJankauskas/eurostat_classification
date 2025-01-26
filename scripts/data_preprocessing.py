import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_for_training(num_workers = 0, add_image_augmentation = False):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if add_image_augmentation:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    script_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(os.path.dirname(script_dir), "dataset")

    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader