import os
from PIL import Image
import matplotlib.pyplot as plt

data_classes = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

def show_data_example():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(os.path.dirname(script_dir), "dataset")

    fig, axes = plt.subplots(5, 2, figsize=(5, 5))
    axes = axes.flatten()

    for i, data_class in enumerate(data_classes):
        sample_image_path = os.path.join(dataset_path, data_class,
                                         os.listdir(os.path.join(dataset_path, data_class))[0])
        img = Image.open(sample_image_path)
        axes[i].imshow(img)
        axes[i].set_title(data_class)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def show_data_distribution():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(os.path.dirname(script_dir), "dataset")

    class_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in data_classes}

    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.title("Img Count In Each Class")
    plt.show()