# ECG DataSet Definition

# Dependencies
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import torch
from os import listdir
from os.path import isfile, join

# Europa Data Set
class ECGDataset(Dataset):
	def __init__(self, image_dir, pytorch=True):
		super().__init__()

		self.files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

		self.pytorch = pytorch

	def __len__(self):
		return len(self.files)

	def map_files(self, ecg_file: Path):
		files = {
			'ecg': ecg_file,
		}
		return files
	
	def open_as_array(self, idx):
		raw_img = np.array(Image.open(self.files[idx]['ecg']))
		normalized_img = raw_img / 255.0
		return normalized_img[None, :, :]

	def __getitem__(self, idx):
		x = torch.tensor(self.open_as_array(idx), dtype=torch.float32)
		return x

	def __repr__(self):
		return 'Dataset class with {} files'.format(self.__len__())