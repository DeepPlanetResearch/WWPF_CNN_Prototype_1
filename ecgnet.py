# E-Net Model definition
import torch
from torch import nn
import numpy as np

class ECGNet(nn.Module):
	def __init__(self, in_channels, out_channels, img_width, img_height, filters = [32, 64, 128, 256]):
		super().__init__()

		self.network = nn.Sequential(
			nn.Conv2d(in_channels, filters[0], kernel_size = 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(filters[0], filters[1], kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(),
			nn.MaxPool2d(2,2),

			nn.Conv2d(filters[1], filters[2], kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(),
			nn.Conv2d(filters[2], filters[2], kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(),
			nn.MaxPool2d(2,2),

			nn.Conv2d(filters[2], filters[3], kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(),
			nn.Conv2d(filters[3], filters[3], kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(),
			nn.MaxPool2d(2,2),

			nn.Flatten(),
			nn.Linear(img_width * img_height, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, out_channels)
		)
		

	def __call__(self, x):
		self.network(x)
