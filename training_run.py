### ENERGI Training Run Generator

# E-Nety Dependencies
from ecg_tensorboard import ECGBoard

# Dependencies
import yaml
import os
import shutil
import torch
import time
from torch import nn
from IPython.display import clear_output
import matplotlib.pyplot as plt

class TrainingRun:
	def __init__(self, config_file):
		print("Loading Configuration File " + config_file)
		self.train_loss = []
		self.valid_loss = []
		self.best_loss = 0.0
		self.training_dir = ""
		self.weights_dir = ""
		self.config_file = config_file
		self.cuda_available = torch.cuda.is_available()

		with open(config_file) as file:
			# Load config file
			self.config = yaml.load(file, Loader = yaml.FullLoader)

	def __deep_supervised_loss(self, outputs, target, loss_function, phase):
		deep_supervision = type(outputs) is tuple
		if (deep_supervision):
			summed_loss = 0.0

			for index, output in enumerate(outputs):
				loss = loss_function(output, target)
				summed_loss += loss.item()
				if phase == 'train':
					if index == (len(outputs) - 1):
						loss.backward()
					else:
						loss.backward(create_graph = True)
			return summed_loss
		else:
			loss = loss_function(outputs, target)
			loss.backward()
			return loss

	def __deep_supervised_accuracy_function(self, outputs, target):
		deep_supervision = type(outputs) is tuple
		target_val = target.cuda() if self.cuda_available else target
		if (deep_supervision):
			summed_accuracy = 0.0
			for output in outputs:
				summed_accuracy += (output.argmax(dim=1) == target_val).float().mean()
			return summed_accuracy / len(outputs)
		else:
			return (outputs.argmax(dim=1) == target_val).float().mean()

	def __train(
			self,
			model,
			train_dataloader,
			validation_dataloader,
			loss_function,
			optimizer,
			enet_board,
			epochs
		):
		start = time.time()
		if self.cuda_available:
			model.cuda()

		for epoch in range(epochs):
			print("Training Started")

			# Timing
			epoch_start = time.time()
			print('Epoch {}/{}'.format(epoch, epochs - 1))
			print('-' * 10)

			for phase in ['train', 'valid']:
				if phase == 'train':
					model.train(True)
					dataloader = train_dataloader
				else:
					model.train(False)
					dataloader = validation_dataloader

				running_loss = 0.0
				running_acc = 0.0
				step = 0

				# iterate over data
				for x, y in dataloader:
					x = x.cuda() if self.cuda_available else x
					y = y.cuda() if self.cuda_available else y
					step += 1

					# forward pass
					if phase == 'train':
						optimizer.zero_grad()
						outputs = model(x)
						loss = self.__deep_supervised_loss(outputs, y, loss_function, phase)
						optimizer.step()

					else:
						with torch.no_grad():
							outputs = model(x)
							loss = self.__deep_supervised_loss(outputs, y.long(), loss_function, phase)

					acc = self.__deep_supervised_accuracy_function(outputs, y)

					running_acc  += acc * dataloader.batch_size
					running_loss += loss * dataloader.batch_size

					if step % 100 == 0:
						clear_output(wait=True)
						memory_allocated = torch.cuda.memory_allocated() if self.cuda_available else 0
						print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, memory_allocated/1024/1024))
					
					if self.cuda_available:
						torch.cuda.empty_cache() 

				epoch_loss = running_loss / len(dataloader.dataset)
				epoch_acc = running_acc / len(dataloader.dataset)

				print('=' * 10)
				if (epoch_loss < self.best_loss):
					print('Loss improved from {} to {}. Saving Weights.'.format(self.best_loss, epoch_loss))
					# Update best Loss
					self.best_loss = epoch_loss
					# Save weights
					torch.save(model.state_dict(), self.weights_dir + 'best.pt')
					
				clear_output(wait=True)
				epoch_elapsed = time.time() - epoch_start
				print('-' * 10)
				print('Epoch {}/{}'.format(epoch, epochs - 1))
				print('-' * 10)
				print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
				print('-' * 10)
				print('Epoch Training time: {:.0f}m {:.0f}s'.format(epoch_elapsed // 60, epoch_elapsed % 60))
				print('=' * 10)

				# Update training and validation losses
				if phase=='train':
					self.train_loss.append(epoch_loss)
					enet_board.update_scalar('Training/Loss', epoch_loss, epoch)
				else:
					self.valid_loss.append(epoch_loss)
					enet_board.update_scalar('Validation/Loss', epoch_loss, epoch)

				# Save Plot results
				plt.figure(figsize=(10,8))
				plt.plot(self.train_loss, label='Train loss')
				plt.plot(self.valid_loss, label='Valid loss')
				plt.legend()
				plt.savefig(self.training_dir + "learning_curve_graph.png")	
				plt.close()

				# Save weights
				torch.save(model.state_dict(), self.weights_dir + 'epoch_' + str(epoch) + '.pt')

		time_elapsed = time.time() - start
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))	
		print('Lowest Validation Loss Achieved: {}'.format(self.best_loss))
		
		return self.train_loss, self.valid_loss

	def start(self, model, train_dataloader, validation_dataloader):
		# Scan trainings dir for existing trainingsloss_function
		dirs = os.listdir(self.config["training"]["trainings_dir"])
		number_of_trainings = len(dirs)
		
		# increment and make new dir
		new_dir_number = number_of_trainings + 1
		self.training_dir = self.config["training"]["trainings_dir"] + str(new_dir_number) + "/"
		print("Creating new training: " + str(new_dir_number))
		os.mkdir(self.training_dir)		

		# make weights dir
		self.weights_dir = self.training_dir + "weights/"
		os.mkdir(self.weights_dir)

		# copy over config file
		shutil.copy(self.config_file, self.training_dir + self.config_file)

		# Set learning rate for Adam
		optimizer = torch.optim.Adam(model.parameters(), lr = self.config["training"]["learning_rate"])

		# Set Loss Function
		if self.cuda_available:
			loss_function = nn.CrossEntropyLoss().cuda()
		else:
			loss_function = nn.CrossEntropyLoss()

		# Tensorboard
		ecg_board = ECGBoard(self.config, self.training_dir)
		ecg_board.launch(model, train_dataloader)
		# enet_board.plot_model_weights(model, 0, 0)
		
		# return with values of various target dirs
		return self.__train(
			model,
			train_dataloader,
			validation_dataloader,
			loss_function,
			optimizer,
			ecg_board,
			epochs=self.config["training"]["epochs"],
		)