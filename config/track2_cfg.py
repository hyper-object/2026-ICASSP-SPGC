import numpy as np 

# data parameters
crop_size = 128 #320

# data info
bands = np.arange(400, 1000, 10)

# training parameters
batch_size = 2
epochs = 14
init_lr = 4e-4