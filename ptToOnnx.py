import torch
import torch.onnx

# Load your PyTorch model
checkpoint = torch.load("OmniParser/weights/icon_detect/best.pt")
model = checkpoint['model'] if 'model' in checkpoint else checkpoint
model.eval()

#print(model)

# Dummy input for the model
dummy_input = torch.randn(1, 3, 640, 640) #80, 512) #, None or 80, 512) #1, 3, 224, 224)  # Adjust as per your input shape

# Export to ONNX
torch.onnx.export(model, dummy_input, "OmniParser/weights/icon_detect/omniparsermodel.onnx", input_names=['input'], output_names=['output'])
