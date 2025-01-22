import torch
from torchsummary import summary
from models.CNN import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_model_summary():
    model = CNN()
    model = model.to(device)

    summary(model, (3, 64, 64), device=device.type)