# Training Program for ENERGI U-Net (E-Net)

# ENERGI Dependencies
from ecgnet import ECGNet
from ecg_dataset import ECGDataset
from training_run import TrainingRun

# Other Dependencies
from torchsummary import summary
from pathlib import Path
import torch
from torch.utils.data import DataLoader


### Program
# ================================
print("Who We Play For (WWPF) ECG Autonomous Diagnosis Neural Network\n\n")
print("Authors: Scott Hasbrouck, (please add your name when you contribute)\n\n")
print("version 0.1.0\n")
print("----------------------\n\n")


# === Instantiate training run ===
training_run = TrainingRun("config.yaml")

# === Load Data ===
data = ECGDataset(
	Path(training_run.config["images"]["base_path"]),
)

# Calculate Data Splits
training_split = training_run.config["images"]["training_split"]
validation_split = training_run.config["images"]["validation_split"]
test_split = len(data) - training_split - validation_split

# Randomly Split Data
training_data, validation_data, test_data = torch.utils.data.random_split(
	data,
	(training_split, validation_split, test_split)
)

print("Found " + str(len(data)) + " Images.\n")

print("Splitting into "
	+ str(training_split,)
	+ " training, "
	+ str(validation_split)
	+ " validation, and "
	+ str(test_split)
	+ " test images.\n")

batch_size = training_run.config["training"]["batch_size"]

training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_data, batch_size=test_split, shuffle=False)

target_labels = training_run.config["target_classes"]

# === Instantiate Model and Print Summary ===
ecg_net = ECGNet(
	3,
	len(target_labels),
	training_run.config["images"]["width"],
	training_run.config["images"]["height"]
)
if torch.cuda.is_available():
	ecg_net.to("cuda")
summary(ecg_net, (1, training_run.config["images"]["width"], training_run.config["images"]["height"]))

print("ECG CNN Model Loaded!\n")
print("*" * 10)

# === Start Training Run ===
training_loss, validation_loss = training_run.start(
	ecg_net,
	training_data_loader,
	validation_data_loader
)