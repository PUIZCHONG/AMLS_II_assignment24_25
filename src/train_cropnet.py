import tensorflow as tf
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

def load_model_with_tfsm(model_path, input_shape=(224, 224, 3)):
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
        
        print("Model loaded successfully with TFSMLayer!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for the model
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Adjust preprocessor based on your model
    return img, x

def find_target_layer(model):
    """
    Find the last convolutional layer in the model
    """
    for layer in reversed(model.layers):
        # Check if the layer has 'conv' in its name
        if 'conv' in layer.name.lower():
            return layer.name
        # For TFSMLayer, we might need to look inside the layer
        elif isinstance(layer, TFSMLayer):
            # This is more complex as TFSMLayer encapsulates the entire SavedModel
            # For this case, we might need to use a predetermined name
            return None
    return None

def make_gradcam_heatmap(img_array, model, class_idx=None):
    """
    Create a Grad-CAM heatmap handling dictionary outputs
    """
    # For TFSMLayer, we need a different approach since we can't easily extract internal layers
    
    # We'll use the gradient of the predicted class with respect to the input image
    with tf.GradientTape() as tape:
        # Cast the image to float32 and make it trainable
        img_array = tf.cast(img_array, tf.float32)
        tape.watch(img_array)
        
        # Get model predictions
        preds = model(img_array, training=False)
        
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
        
        # If class_idx is None, use the predicted class
        if class_idx is None:
            class_idx = tf.argmax(prediction_values[0])
        
        # Get the score for the target class
        target_class_score = prediction_values[0, class_idx]
    
    # Calculate gradients with respect to the input image
    gradients = tape.gradient(target_class_score, img_array)
    
    # Global average pooling for the gradients
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    
    # Create a weighted combination of the feature maps using the gradients
    # For this approach, we're essentially creating the heatmap directly from the input
    # This is a simplified approach that won't be as precise as using the last conv layer
    heatmap = tf.reduce_mean(tf.abs(gradients[0]), axis=-1)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, output_dir, class_names=None, pred_class=None, 
                            pred_score=None, alpha=0.4):
    """
    Save and display Grad-CAM visualization
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the superimposed image to the specified output directory
    image_filename = os.path.basename(img_path)
    cam_path = os.path.join(output_dir, f"gradcam_{image_filename}")
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    # Create title with prediction class and score if available
    title = "Grad-CAM"
    if class_names is not None and pred_class is not None and pred_score is not None:
        class_name = class_names[pred_class]
        title = f"{class_name} ({pred_score:.2f})"
    
    # Display the original image and the Grad-CAM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(superimposed_img)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Grad-CAM saved to {cam_path}")

def apply_gradcam(model_path, img_path, class_names, output_dir):
    """
    Apply Grad-CAM visualization on an image using the specified model
    
    Args:
        model_path: Path to the saved model directory
        img_path: Path to the image file
        class_names: List of class names for the model
        output_dir: Directory to save Grad-CAM images
    """
    # Load model
    model = load_model_with_tfsm(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Preprocess image
    original_img, preprocessed_img = preprocess_image(img_path)
    
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
    
    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(preprocessed_img, model, pred_class)
    
    # Save and display the Grad-CAM visualization
    save_and_display_gradcam(img_path, heatmap, output_dir, class_names, pred_class, pred_score)
    
    return pred_class, pred_score

# Example usage
if __name__ == "__main__":
    # Cassava disease class names
    class_names = [
        "Cassava Bacterial Blight (CBB)",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mite (CGM)",
        "Cassava Mosaic Disease (CMD)",
        "Healthy"
    ]
    
    # Your specific image path
    image_path = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\cassava-leaf-disease-classification\train_images\3213974381.jpg"
    
    # Path to your model directory (not the .pb file directly)
    model_path = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\models"
    
    # Output directory for saving Grad-CAM images
    output_dir = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\images"
    
    # Apply GradCAM
    apply_gradcam(model_path, image_path, class_names, output_dir)