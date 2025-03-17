import onnxruntime as ort
import numpy as np

# Set up the ONNX Runtime session
ort_session = ort.InferenceSession("car_rim_classifier.onnx")

# Get the correct input name from the model (after verifying in the previous step)
input_name = "input.1"  # Example input name

# Create a dummy input (simulating an image)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference with the correct input name
outputs = ort_session.run(None, {input_name: dummy_input})

# Print the inference result
print("Inference Output:", outputs)
