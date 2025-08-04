import torch

BATCH_SIZE = 32
NUM_CLASSES = 12  # Based on your image (10 folders)
IMG_SIZE = 224
EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda")

DATA_DIR = "data2"
MODEL_PATH = "resnet50_cloud_finetuned.pth"
CLASS_NAMES = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cl', 'Cs', 'Ct', 'Cu', 'Ns', 'Sc', 'St']
