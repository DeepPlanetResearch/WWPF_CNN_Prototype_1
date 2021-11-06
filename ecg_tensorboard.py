# Tensorboard Setup for E-Net

# Dependencies
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

class ECGBoard():
	def __init__(self, config, training_dir):
		super().__init__()

		log_dir = training_dir + config["tensorboard"]["subdir"]

		self.writer = SummaryWriter(log_dir)
		self.tb = program.TensorBoard()

		self.tb.configure(argv=[None, '--logdir', log_dir])

	def __plot_pyplt_to_tensorboard(self, fig, step, label):
		"""
		Takes a matplotlib figure handle and converts it using
		canvas and string-casts to a numpy array that can be
		visualized in TensorBoard using the add_image function

		Parameters:
		writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
		fig (matplotlib.pyplot.fig): Matplotlib figure handle.
		step (int): counter usually specifying steps/epochs/time.
		"""

		# Draw figure on canvas
		fig.canvas.draw()

		# Convert the figure to numpy array, read the pixel values and reshape the array
		img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

		# Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
		img = img / 255.0
		# img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

		# Add figure in numpy "image" to TensorBoard writer
		self.writer.add_image(label, img, step)
		plt.close(fig)

	def __plot_filters_single_channel_big(self, t, epoch, labels):
		#setting the rows and columns
		nrows = t.shape[0]*t.shape[2]
		ncols = t.shape[1]*t.shape[3]


		npimg = np.array(t.numpy(), np.float32)
		npimg = npimg.transpose((0, 2, 1, 3))
		npimg = npimg.ravel().reshape(nrows, ncols)

		npimg = npimg.T

		fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))  
		imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)

		figure = imgplot.get_figure()

		self.__plot_pyplt_to_tensorboard(fig, epoch, labels)

	def __plot_filters_single_channel(self, t, epoch, label):
		#kernels depth * number of kernels
		nplots = t.shape[0]*t.shape[1]
		ncols = 12

		nrows = 1 + nplots//ncols
		#convert tensor to numpy image
		npimg = np.array(t.numpy(), np.float32)

		count = 0
		fig = plt.figure(figsize=(ncols, nrows))

		#looping through all the kernels in each channel
		for i in range(t.shape[0]):
			for j in range(t.shape[1]):
				count += 1
				ax1 = fig.add_subplot(nrows, ncols, count)
				npimg = np.array(t[i, j].numpy(), np.float32)
				npimg = (npimg - np.mean(npimg)) / np.std(npimg)
				npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
				ax1.imshow(npimg)
				ax1.set_title(str(i) + ',' + str(j))
				ax1.axis('off')
				ax1.set_xticklabels([])
				ax1.set_yticklabels([])

		plt.tight_layout()

		self.__plot_pyplt_to_tensorboard(fig, epoch, label)

	def __plot_filters_multi_channel(self, t):
		#get the number of kernals
		num_kernels = t.shape[0]    

		#define number of columns for subplots
		num_cols = 12
		#rows = num of kernels
		num_rows = num_kernels

		#set the figure size
		fig = plt.figure(figsize=(num_cols,num_rows))

		#looping through all the kernels
		for i in range(t.shape[0]):
			ax1 = fig.add_subplot(num_rows,num_cols,i+1)
			
			#for each kernel, we convert the tensor to numpy 
			npimg = np.array(t[i].numpy(), np.float32)
			#standardize the numpy image
			npimg = (npimg - np.mean(npimg)) / np.std(npimg)
			npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
			npimg = npimg.transpose((1, 2, 0))
			ax1.imshow(npimg)
			ax1.axis('off')
			ax1.set_title(str(i))
			ax1.set_xticklabels([])
			ax1.set_yticklabels([])
			
		plt.tight_layout()

		self.__plot_pyplt_to_tensorboard(fig, epoch, label)

	def launch(
		self,
		model,
		training_dataloader
	):
		self.url = self.tb.launch()

		dataiter = iter(training_dataloader)
		images, labels = dataiter.next()

		img_grid = torchvision.utils.make_grid(images)
		self.writer.add_image('Batch of Training ECGs', img_grid)

		self.writer.add_graph(model.to("cpu"), images)
		self.writer.flush()

		print(("*" * 10) + "\n\nTensorboard Running on: " + self.url + "\n\n" + ("*" * 10))
		return self.url

	def update_scalar(self, label, value, epoch):
		self.writer.add_scalar(label, value, epoch)
		self.writer.flush()

	def plot_model_weights(self, model, epoch, single_channel = True, collated = False):
		# Get all Conv2D Layers from model

		# Pickup here
		for layer in model.conv0_0.modules():
			print(layer)
			if isinstance(layer, nn.Conv2d):
				if single_channel:
					if collated:
						self.__plot_filters_single_channel_big(layer.weight.data, epoch)
					else:
						self.__plot_filters_single_channel(layer.weight.data, epoch)
				else:
					if layer.weight.shape[1] == 3:
						self.__plot_filters_multi_channel(layer.weight.data)
					else:
						print("Can only plot weights with three channels with single channel = False")