import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define the class labels
class_labels = ["car_side", "rim", "wrong"]

# Load the trained model
model = models.resnet18(pretrained=False)  # Initialize a ResNet model
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Same structure as training

# Load the trained weights
model.load_state_dict(torch.load("car_rim_classifier.pth"))
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the training images
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization used in training
])

# Load and preprocess a new image
image_path = "test_images/test2.jpg"  # Replace with your test image path
image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
image = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

print(f"Predicted Class: {class_labels[predicted_class]} | Probabilities: {probabilities.tolist()}")

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class index
predicted_class = torch.argmax(probabilities).item()

# Print the results
print(f"Predicted Class: {class_labels[predicted_class]}")
print(f"Confidence Scores: {probabilities.tolist()}")