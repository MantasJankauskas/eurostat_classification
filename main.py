from utils.TrainedModelStatistics import TrainedModelStatistics
from utils.dataset_importer import download_dataset
from utils.data_preview import show_data_example, show_data_distribution
from utils.cnn_summary import show_model_summary
from scripts.data_preprocessing import get_data_for_training
from scripts.train_model import train_model, test_model
from models.CNN import CNN

if __name__ == "__main__":
    # download_dataset()

    # data statistics
    # show_data_example()
    # show_data_distribution()

    # show_model_summary()

    #train and test model
    model = CNN()
    model_training_statistics = TrainedModelStatistics(model=model)
    train_loader, val_loader, test_loader = get_data_for_training(num_workers=2, add_image_augmentation=True)
    trained_model = train_model(model, model_training_statistics, train_loader, val_loader, epochs=1, show_info=True)

    # test_model(trained_model, test_loader)

    model_training_statistics.show_confusion_matrix(test_loader=test_loader)
    model_training_statistics.plot_training_graphs()