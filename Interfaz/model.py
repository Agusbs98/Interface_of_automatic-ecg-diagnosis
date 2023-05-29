import torch
import torch.nn as nn
from torchsummary import summary

model = torch.load("./models/CPSC-2018/best.ptl", map_location = "cpu")

summary(model, input_size = (3,5000))