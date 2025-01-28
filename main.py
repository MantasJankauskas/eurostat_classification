from models.MobileNetV2 import MobileNetV2
from utils.TrainedModelStatistics import TrainedModelStatistics
from utils.dataset_importer import download_dataset
from utils.data_preview import show_data_example, show_data_distribution
from utils.cnn_summary import show_model_summary
from scripts.data_preprocessing import get_data_for_training
from scripts.train_model import train_model, test_model
from models.MyCNN import MyCNN

if __name__ == "__main__":
    # download_dataset()

    # data statistics
    # show_data_example()
    # show_data_distribution()

    # model = MyCNN()
    model = MobileNetV2()

    # show_model_summary(model)

    #train and test model
    model_training_statistics = TrainedModelStatistics(model=model)
    train_loader, val_loader, test_loader = get_data_for_training(num_workers=3, add_image_augmentation=True)
    trained_model = train_model(model, model_training_statistics, train_loader, val_loader, epochs=30, show_info=True)

    test_model(trained_model, test_loader)

    model_training_statistics.show_confusion_matrix(test_loader=test_loader)
    model_training_statistics.plot_training_graphs()