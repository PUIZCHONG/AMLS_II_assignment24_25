import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import for TensorFlow model
import tensorflow as tf
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model

# Imports for PyTorch model
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm  # Make sure to install this package if not already installed

# Set environment variable to avoid duplicate lib warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Paths to models and test image
TF_MODEL_PATH = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\models"
TORCH_MODEL_PATH = r"models/tf_efficientnet_b4.ns_jft_in1k_fold_0_epoch_14_checkpoint.pth"
TEST_IMAGE_PATH = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\cassava-leaf-disease-classification\test_images\2216849948.jpg"
OUTPUT_DIR = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\images"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names
CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

# Model weights (optional - adjust if one model performs better)
# Default is equal weight [0.5, 0.5]
MODEL_WEIGHTS = [0.5, 0.5]

# ========== TensorFlow Model Functions ==========

def load_tf_model(model_path, input_shape=(224, 224, 3)):
    """
    Load a saved TensorFlow model using TFSMLayer
    """
    try:
        # Create TFSMLayer
        layer = TFSMLayer(model_path, call_endpoint='serving_default')
        
        # Wrap TFSMLayer in a new model for prediction
        input_layer = Input(shape=input_shape)
        output_layer = layer(input_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        
        print("TensorFlow model loaded successfully with TFSMLayer!")
        return model
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        return None

def preprocess_image_tf(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for the TensorFlow model
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Adjust preprocessor based on your model
    return img, x

def tf_model_predict(model, img_path):
    """
    Make prediction using the TensorFlow model - following the pattern from your code
    """
    # Preprocess image
    _, preprocessed_img = preprocess_image_tf(img_path)
    
    # Make predictions
    preds = model.predict(preprocessed_img)
    
    # Handle dictionary output if needed
    if isinstance(preds, dict):
        # Find the key that contains the predictions
        possible_keys = ['predictions', 'outputs', 'output', 'logits', 'probs']
        pred_key = None
        
        for key in possible_keys:
            if key in preds:
                pred_key = key
                break
                
        if pred_key is None:
            # If none of the common keys match, use the first key
            pred_key = list(preds.keys())[0]
            
        prediction_values = preds[pred_key]
    else:
        prediction_values = preds
    
    # Get predicted class and score
    pred_class = np.argmax(prediction_values[0])
    pred_score = float(prediction_values[0, pred_class])
    
    # Normalize probabilities to sum to 1
    probs = prediction_values[0]
    if probs.sum() > 0:
        probs = probs / probs.sum()
    
    return pred_class, pred_score, probs

# ========== PyTorch Model Functions ==========

class CustomEfficientNet(torch.nn.Module):
    """PyTorch EfficientNet model from checkpoint - exactly as in your file"""
    def __init__(self, checkpoint_path, num_classes=5):
        super(CustomEfficientNet, self).__init__()
        
        # Load checkpoint
        print(f"Attempting to load PyTorch model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create a backbone model - try to match the architecture from the state dict keys
        if any(k.startswith('model.blocks') for k in checkpoint['model_state_dict'].keys()):
            # Using a timm model as the base which is more flexible
            self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
            
            # Manually map keys if needed - this may need further customization
            state_dict = checkpoint['model_state_dict']
            
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

def load_torch_model_simplified(num_classes=5):
    """Load just a pretrained EfficientNet model without checkpoint"""
    try:
        # This creates an EfficientNet with ImageNet weights
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        model.eval()
        print("PyTorch model loaded using pretrained weights (not from checkpoint)")
        return model
    except Exception as e:
        print(f"Error loading simplified PyTorch model: {e}")
        return None

def load_torch_model(model_path, num_classes=5):
    """Load the PyTorch model following your implementation"""
    try:
        # Try loading the custom model from checkpoint
        model = CustomEfficientNet(model_path, num_classes=num_classes)
        model.eval()
        print("PyTorch model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading PyTorch model from checkpoint: {e}")
        print("Falling back to pretrained model...")
        return load_torch_model_simplified(num_classes)

def torch_model_predict(model, img_path):
    """
    Make prediction using the PyTorch model - exactly as in your file
    """
    # Load and preprocess image - match the preprocessing in your file
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    # Get model prediction - match the prediction logic in your file
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Get prediction score
    pred_score = float(probabilities[0, predicted_class])
    
    # Convert to numpy for consistency
    probs = probabilities[0].numpy()
    
    return predicted_class, pred_score, probs

# ========== Ensemble Function ==========

def ensemble_predict(tf_model, torch_model, img_path, weights=None):
    """
    Make ensemble prediction using both models
    
    Args:
        tf_model: TensorFlow model
        torch_model: PyTorch model
        img_path: Path to image
        weights: List of weights [tf_weight, torch_weight]
    
    Returns:
        Dictionary with prediction results
    """
    # Set default weights if not provided
    if weights is None:
        weights = [0.5, 0.5]
    
    # Normalize weights to sum to 1
    weights = [w / sum(weights) for w in weights]
    
    # Make predictions with each model
    tf_class, tf_score, tf_probs = tf_model_predict(tf_model, img_path)
    torch_class, torch_score, torch_probs = torch_model_predict(torch_model, img_path)
    
    # Combine predictions
    ensemble_probs = (tf_probs * weights[0]) + (torch_probs * weights[1])
    ensemble_class = np.argmax(ensemble_probs)
    ensemble_score = float(ensemble_probs[ensemble_class])
    
    # Return all results
    return {
        'ensemble': {
            'class_idx': int(ensemble_class),
            'class_name': CLASS_NAMES[ensemble_class],
            'probability': ensemble_score,
            'probabilities': ensemble_probs.tolist()
        },
        'CropNet': {
            'class_idx': int(tf_class),
            'class_name': CLASS_NAMES[tf_class],
            'probability': float(tf_score),
            'probabilities': tf_probs.tolist()
        },
        'EfficientNetB4': {
            'class_idx': int(torch_class),
            'class_name': CLASS_NAMES[torch_class],
            'probability': float(torch_score),
            'probabilities': torch_probs.tolist()
        }
    }

def visualize_ensemble(results, img_path, output_path):
    """Create visualization of ensemble results"""
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # Define grid layout
    gs = plt.GridSpec(2, 4, figure=fig)
    
    # Image plot
    ax_img = fig.add_subplot(gs[0, 0:2])
    ax_img.imshow(img)
    ax_img.set_title('Original Image')
    ax_img.axis('off')
    
    # Results text area
    ax_text = fig.add_subplot(gs[0, 2:4])
    ax_text.axis('off')
    
    # Add results text
    ensemble_result = results['ensemble']
    tf_result = results['CropNet']
    torch_result = results['EfficientNetB4']
    
    result_text = f"ENSEMBLE PREDICTION\n"
    result_text += f"Class: {ensemble_result['class_name']}\n"
    result_text += f"Confidence: {ensemble_result['probability']:.4f}\n\n"
    
    result_text += f"MODEL PREDICTIONS\n"
    result_text += f"CropNet: {tf_result['class_name']} ({tf_result['probability']:.4f})\n"
    result_text += f"EfficientNetB4: {torch_result['class_name']} ({torch_result['probability']:.4f})\n\n"
    
    result_text += f"WEIGHTS\n"
    result_text += f"CropNet: {MODEL_WEIGHTS[0]:.2f}, EfficientNetB4: {MODEL_WEIGHTS[1]:.2f}"
    
    ax_text.text(0.05, 0.95, result_text, transform=ax_text.transAxes, 
                 fontsize=12, va='top', family='monospace')
    
    # Bar charts for each model
    models = ['CropNet', 'EfficientNetB4', 'ensemble']
    colors = ['#36A2EB', '#FF6384', '#4BC0C0']
    
    for i, (model_name, color) in enumerate(zip(models, colors)):
        # Use the exact key for each model, without converting to lowercase
        model_key = model_name
        probs = results[model_key]['probabilities']
        
        ax = fig.add_subplot(gs[1, i])
        ax.bar(range(len(CLASS_NAMES)), probs, color=color, alpha=0.7)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([f"Class {i}" for i in range(len(CLASS_NAMES))], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name} Prediction')
    
    # Add detailed legend for class names
    ax_legend = fig.add_subplot(gs[1, 3])
    ax_legend.axis('off')
    
    legend_text = "CLASS LEGEND\n\n"
    for i, name in enumerate(CLASS_NAMES):
        legend_text += f"Class {i}: {name}\n"
    
    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                  fontsize=10, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

def print_results(results):
    """Print detailed results to console"""
    print("\n" + "="*50)
    print("ENSEMBLE PREDICTION RESULTS")
    print("="*50)
    
    print(f"\nENSEMBLE PREDICTION:")
    print(f"  Class: {results['ensemble']['class_name']}")
    print(f"  Confidence: {results['ensemble']['probability']:.4f}")
    
    print(f"\nINDIVIDUAL MODEL PREDICTIONS:")
    print(f"  CropNet: {results['CropNet']['class_name']} "
          f"(Confidence: {results['CropNet']['probability']:.4f})")
    print(f"  EfficientNetB4: {results['EfficientNetB4']['class_name']} "
          f"(Confidence: {results['EfficientNetB4']['probability']:.4f})")
    
    print("\nDETAILED PROBABILITY TABLE:")
    header = "Class".ljust(30) + "CropNet".ljust(15) + "EfficientNetB4".ljust(15) + "Ensemble".ljust(15)
    print(header)
    print("-" * len(header))
    
    for i, class_name in enumerate(CLASS_NAMES):
        tf_prob = results['CropNet']['probabilities'][i]
        torch_prob = results['EfficientNetB4']['probabilities'][i]
        ensemble_prob = results['ensemble']['probabilities'][i]
        
        print(f"{class_name.ljust(30)}{tf_prob:.4f}".ljust(45) + 
              f"{torch_prob:.4f}".ljust(15) + f"{ensemble_prob:.4f}")
    
    print("="*50)


# ========== Main Execution ==========

def main():
    print("Loading models...")
    
    # Load TensorFlow model
    tf_model = load_tf_model(TF_MODEL_PATH)
    if tf_model is None:
        print("Failed to load TensorFlow model. Exiting.")
        return
    
    # Load PyTorch model using the pattern from your file
    torch_model = load_torch_model(TORCH_MODEL_PATH, num_classes=len(CLASS_NAMES))
    if torch_model is None:
        print("Failed to load PyTorch model. Using TensorFlow model only.")
        MODEL_WEIGHTS[0] = 1.0
        MODEL_WEIGHTS[1] = 0.0
    
    print("\nMaking ensemble prediction...")
    
    # Make ensemble prediction
    results = ensemble_predict(tf_model, torch_model, TEST_IMAGE_PATH, MODEL_WEIGHTS)
    
    # Print results
    print_results(results)
    
    # Visualize results
    output_path = os.path.join(OUTPUT_DIR, "ensemble_prediction.png")
    visualize_ensemble(results, TEST_IMAGE_PATH, output_path)
    
    print("\nEnsemble prediction complete!")

if __name__ == "__main__":
    main()