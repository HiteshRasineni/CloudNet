import torch

BATCH_SIZE = 32
NUM_CLASSES = 10  # Based on your image (10 folders)
IMG_SIZE = 224
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
MODEL_PATH = "resnet50_cloud_finetuned.pth"
CLASS_NAMES = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus',
               'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus']