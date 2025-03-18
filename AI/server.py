from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model architecture exactly as used during training
class CarRimClassifier(nn.Module):
    def __init__(self):
        super(CarRimClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)  # 3 output classes: car_side, rim, wrong
    
    def forward(self, x):
        return self.model(x)

# Initialize the model
model = CarRimClassifier()

# Load the checkpoint
checkpoint = torch.load("car_rim_classifier.pth", map_location=torch.device("cpu"))

# The error you encountered indicates that your saved checkpoint's keys do not have the "model." prefix.
# We can adjust the checkpoint keys accordingly.
if not all(key.startswith("model.") for key in checkpoint.keys()):
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_checkpoint["model." + key] = value
    checkpoint = new_checkpoint

# Load the adjusted checkpoint into the model
model.load_state_dict(checkpoint)
model.eval()  # Set the model to evaluation mode

# Define the class labels
class_labels = ["car_side", "rim", "wrong"]

# Define image preprocessing (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            outputs = model(image)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class].item()

        # Return the predicted class and confidence
        return {"class": class_labels[predicted_class], "confidence": float(confidence)}
    
    except Exception as e:
        return {"error": str(e)}
