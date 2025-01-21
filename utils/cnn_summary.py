import torch
from torchsummary import summary
from models.CNN import CNN

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, (3, 128, 128), device=device.type)