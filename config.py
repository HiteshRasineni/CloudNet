import torch

BATCH_SIZE = 32
NUM_CLASSES = 12  # Based on your image (10 folders)
IMG_SIZE = 224
EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data2"
MODEL_PATH = "resnet50_cloud_finetuned.pth"
CLASS_NAMES = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cl', 'Cs', 'Ct', 'Cu', 'Ns', 'Sc', 'St']

# Supervised Contrastive Learning (SupCon) hyperparameters
USE_SUPCON = True  # Set to False to use standard cross-entropy training
EMBEDDING_DIM = 128  # Dimension of contrastive learning embeddings
SUPCON_TEMPERATURE = 0.07  # Temperature parameter for contrastive loss
SUPCON_WEIGHT = 0.5  # Weight for contrastive loss (balance with classification loss)