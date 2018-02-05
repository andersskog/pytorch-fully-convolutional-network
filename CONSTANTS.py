import torch

# Define global params for training
BATCH_SIZE = 100
NUM_EPOCHS = 50
NUM_CLASSES = 151

# Detect if Cuda should be used
USE_CUDA = torch.cuda.is_available()