import torch
from torchsummary import summary
from models.MyCNN import MyCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_model_summary(model):
    model = model.to(device)
    summary(model, (3, 64, 64), device=device.type)