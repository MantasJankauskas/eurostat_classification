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
    train_loader, val_loader, test_loader = get_data_for_training(num_workers=4)
    trained_model = train_model(CNN(), train_loader, val_loader, epochs=1, show_info=False)
    test_model(trained_model, test_loader)


