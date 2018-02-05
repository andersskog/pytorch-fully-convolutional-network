import torch

# Define global params for training
BATCH_SIZE = 1
NUM_EPOCHS = 50
NUM_CLASSES = 3173

# Detect if Cuda should be used
USE_CUDA = torch.cuda.is_available()