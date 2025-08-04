import torch
from torchvision import transforms
from PIL import Image
import os
import config
from model import get_model

# Define the transform
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load model and send to device
device = config.DEVICE
model = get_model()
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()
model.to(device)

def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_name = config.CLASS_NAMES[predicted.item()]
    return class_name

def predict_folder(folder_path, model, device):
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            pred = predict_image(full_path, model, device)
            print(f"{file_name} → {pred}")

# Example usage:
# For single image
print("Prediction:", predict_image("testing/005.jpeg", model, device))

# For folder
# predict_folder("testing", model, device)
