import torch
import torchvision.models as models
import torch.nn as nn

# Load a pre-trained ResNet18 model (excluding the final fully connected layer)
model = models.resnet18(pretrained=True)

# Replace the fully connected layer with one for 3 output classes
model.fc = nn.Linear(model.fc.in_features, 3)

# Load the weights, but skip the 'fc' layer to avoid the mismatch
checkpoint = torch.load('car_rim_classifier.pth')
model_dict = model.state_dict()

# Filter out the fc layer from the checkpoint weights (since it's different)
checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}

# Update the model with the filtered weights
model_dict.update(checkpoint)
model.load_state_dict(model_dict)

# Now the model should be successfully loaded
model.eval()

# You can now export the model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "car_rim_classifier.onnx", opset_version=11)
print("Model exported successfully")
