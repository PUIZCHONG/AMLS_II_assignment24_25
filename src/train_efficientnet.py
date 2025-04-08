import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import timm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_handles = []
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        handle1 = target_layer.register_forward_hook(forward_hook)
        handle2 = target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.append(handle1)
        self.hook_handles.append(handle2)
    
    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)
        
        # Get model output
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.mean([2, 3], keepdim=True)
        activations = self.activations
        
        # Weight the channels by corresponding gradients
        weighted_activations = gradients * activations
        
        # Sum all weighted activations
        cam = torch.sum(weighted_activations, dim=1).squeeze()
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

class CustomEfficientNet(torch.nn.Module):
    def __init__(self, checkpoint_path, num_classes=5):
        super(CustomEfficientNet, self).__init__()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create a backbone model - try to match the architecture from the state dict keys
        if any(k.startswith('model.blocks') for k in checkpoint['model_state_dict'].keys()):
            # Using a timm model as the base which is more flexible
            self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
            
            # Manually map keys if needed - this may need further customization
            state_dict = checkpoint['model_state_dict']
            
            # Initialize a new state dict for the timm model
            new_state_dict = {}
            
            # Print some keys for debugging
            print("Original checkpoint keys (first 5):")
            for i, k in enumerate(list(state_dict.keys())[:5]):
                print(f"  {k}")
            
            print("Target model keys (first 5):")
            for i, k in enumerate(list(self.model.state_dict().keys())[:5]):
                print(f"  {k}")
            
            # Load all weights from the checkpoint that match the model's structure
            try:
                # First, try to load weights using a mapping approach
                model_dict = self.model.state_dict()
                
                # Create a filtered and mapped state dict
                filtered_state_dict = {}
                
                # Print total number of keys for debugging
                print(f"Total keys in checkpoint: {len(state_dict)}")
                print(f"Total keys in model: {len(model_dict)}")
                
                # Map checkpoint keys to model keys - this requires knowledge of the model structure
                for k, v in state_dict.items():
                    # Map custom_head to classifier
                    if k.startswith('custom_head'):
                        if '.1.weight' in k:
                            filtered_state_dict['classifier.weight'] = v
                        elif '.1.bias' in k:
                            filtered_state_dict['classifier.bias'] = v
                    
                    # Map backbone weights - requires knowledge of key patterns
                    elif k.startswith('model.'):
                        # Remove the 'model.' prefix and try to find matching key
                        potential_key = k[6:]  # Strip 'model.' prefix
                        if potential_key in model_dict:
                            filtered_state_dict[potential_key] = v
                
                # Update the model weights with the filtered state dict
                self.model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded {len(filtered_state_dict)} matching weights from checkpoint")
                
                # Fall back to just loading classifier if the above doesn't work
                if len(filtered_state_dict) <= 2 and 'custom_head.1.weight' in state_dict:
                    self.model.classifier.weight.data = state_dict['custom_head.1.weight']
                    self.model.classifier.bias.data = state_dict['custom_head.1.bias']
                    print("Could only load classifier weights from checkpoint")
            except Exception as e:
                print(f"Error during weight loading: {str(e)}")
                # Fallback to just loading the classification head
                if 'custom_head.1.weight' in state_dict:
                    self.model.classifier.weight.data = state_dict['custom_head.1.weight']
                    self.model.classifier.bias.data = state_dict['custom_head.1.bias']
                    print("Falling back to loading only classifier weights from checkpoint")
        else:
            raise ValueError("Could not determine model architecture from checkpoint")
    
    def forward(self, x):
        return self.model(x)

def find_conv_layers(model):
    """
    Find all convolutional layers in the model
    """
    conv_layers = []
    for name, module in model.model.named_modules():
        # Look for convolutional layers (avoid dropout and other layers that might cause issues)
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))
    
    return conv_layers

def visualize_gradcam(image_path, model_path, num_classes=5, class_names=None, layer_name=None):
    """
    Visualize Grad-CAM for a custom EfficientNet model trained on cassava classification
    
    Args:
        image_path: Path to the input image
        model_path: Path to the .pth model file
        num_classes: Number of classes in the model
        class_names: List of class names (optional)
        layer_name: Name of specific layer to target (if None, will use the first conv layer)
    """
    # Load custom model
    try:
        print(f"Attempting to load model from: {model_path}")
        # Check if file exists
        import os
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = CustomEfficientNet(model_path, num_classes=num_classes)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    # Find all convolutional layers
    conv_layers = find_conv_layers(model)
    print(f"Found {len(conv_layers)} convolutional layers")
    
    # Print the first 10 and last 10 convolutional layers for reference
    print("\nFirst 10 convolutional layers:")
    for i, (name, _) in enumerate(conv_layers[:10]):
        print(f"  {i}: {name}")
    
    print("\nLast 10 convolutional layers:")
    for i, (name, _) in enumerate(conv_layers[-10:]):
        print(f"  {len(conv_layers) - 10 + i}: {name}")
    
    # Select target layer
    # If layer_name is provided, find that specific layer
    if layer_name:
        target_layer = None
        for name, module in model.model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
    else:
        # Modified to use the first convolutional layer instead of the 6th from the end
        target_idx = 0  # Select the first layer (index 0)
        target_name, target_layer = conv_layers[target_idx]
    
    print(f"Selected target layer: {target_name}")
    
    # Initialize Grad-CAM with the selected target layer
    grad_cam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Generate Grad-CAM
    cam = grad_cam(input_tensor, predicted_class)
    grad_cam.remove_hooks()  # Clean up hooks
    
    # Convert PIL image to numpy array for visualization
    original_img = np.array(img.resize((380, 380)))
    
    # Create heatmap - OpenCV uses BGR color order
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # Convert BGR to RGB for proper display with matplotlib
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Make sure original_img is in RGB format (PIL loads as RGB, but when converting to numpy it stays RGB)
    # No need to convert original_img since it's already in RGB
    
    # Superimpose heatmap on original image
    # Both heatmap and original_img are now in RGB format
    superimposed = heatmap * 0.4 + original_img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(heatmap)
    ax[1].set_title('Grad-CAM Heatmap (First conv layer)')
    ax[1].axis('off')
    
    ax[2].imshow(superimposed)
    ax[2].set_title('Superimposed')
    ax[2].axis('off')
    
    # Add prediction information
    if class_names:
        pred_class_name = class_names[predicted_class]
        prob_value = probabilities[0, predicted_class].item() * 100
        plt.suptitle(f'Predicted: {pred_class_name} ({prob_value:.2f}%)', fontsize=15)
    else:
        prob_value = probabilities[0, predicted_class].item() * 100
        plt.suptitle(f'Predicted Class: {predicted_class} ({prob_value:.2f}%)', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('gradcam_result_first_layer.png')
    plt.show()


if __name__ == "__main__":
    # Cassava disease class names (adjust these based on your specific classes)
    class_names = [
        "Cassava Bacterial Blight (CBB)",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mite (CGM)",
        "Cassava Mosaic Disease (CMD)",
        "Healthy"
    ]
    
    # Replace with your actual paths
    image_path = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\cassava-leaf-disease-classification\train_images\3213974381.jpg"
    model_path = r"models/tf_efficientnet_b4.ns_jft_in1k_fold_0_epoch_14_checkpoint.pth"
    
    # Run GradCAM visualization focusing on the first convolutional layer
    visualize_gradcam(image_path, model_path, num_classes=5, class_names=class_names)