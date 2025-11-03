import torch

BATCH_SIZE = 32

SUPCON_EPOCHS = 20
FINETUNE_EPOCHS = 10
SUPCON_MODEL_PATH = "supcon_encoder.pth"
FINETUNE_LR = 1e-3

NUM_CLASSES = 12  # Based on your image (10 folders)
IMG_SIZE = 224
EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data2"
MODEL_PATH = "resnet50_cloud_finetuned.pth"
CLASS_NAMES = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cl', 'Cs', 'Ct', 'Cu', 'Ns', 'Sc', 'St']
