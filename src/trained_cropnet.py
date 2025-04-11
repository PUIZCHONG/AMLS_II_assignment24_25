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

def make_gradcam_heatmap_for_last_layer(img_array, model, class_idx=None, threshold=0.2):
    """
    Create a Grad-CAM heatmap focusing on the last layer
    """
    # For TFSMLayer models, we'll focus on the last layer by using gradients from prediction
    try:
        # For TFSMLayer, we need a different approach since we can't easily extract internal layers
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
        
        # This is a crucial change from the first layer approach
        # For last layer visualization, we need to use the gradient magnitude
        # to create a heatmap of where the model is focusing for this particular class
        
        # Global average pooling for the gradients across the channels
        # This creates an aggregated representation of importance
        pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
        
        # Create a weighted combination using the average gradients
        # This simulates the final layer's attention
        weighted_input = tf.reduce_mean(tf.abs(gradients[0]), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(weighted_input, 0) / tf.math.reduce_max(weighted_input)
        
        # Apply threshold to enhance contrast
        heatmap = tf.where(heatmap < threshold, 0.0, heatmap)
        
        # Re-normalize after thresholding
        if tf.math.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), int(class_idx.numpy()), float(target_class_score.numpy())
    
    except Exception as e:
        print(f"Error in gradient calculation: {e}")
        return None, None, None

def visualize_last_layer_with_occlusion(model, img_path, output_dir, class_names=None):
    """
    Visualize last layer importance using occlusion sensitivity
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess image
    original_img, preprocessed_img = preprocess_image(img_path)
    
    # Get the base prediction
    preds = model.predict(preprocessed_img, verbose=0)
    
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
    
    # Get the predicted class and score
    pred_class = np.argmax(prediction_values[0])
    pred_score = float(prediction_values[0, pred_class])
    
    # Create occlusion map
    occlusion_map = np.zeros((preprocessed_img.shape[1], preprocessed_img.shape[2]))
    
    # Size of the occlusion patch
    patch_size = 16  # Using a larger patch for efficiency
    
    # Track progress
    total_patches = ((preprocessed_img.shape[1] // patch_size) + 1) * \
                   ((preprocessed_img.shape[2] // patch_size) + 1)
    processed_patches = 0
    
    print(f"Analyzing last layer using occlusion sensitivity...")
    print(f"Base prediction: {class_names[pred_class] if class_names else pred_class} ({pred_score:.4f})")
    print(f"Total patches to analyze: {total_patches}")
    
    # For each position in the image
    for i in range(0, preprocessed_img.shape[1], patch_size):
        for j in range(0, preprocessed_img.shape[2], patch_size):
            # Create a copy of the image with this patch occluded
            occluded_img = preprocessed_img.copy()
            
            # Calculate the effective patch size (handling edge cases)
            i_end = min(i + patch_size, preprocessed_img.shape[1])
            j_end = min(j + patch_size, preprocessed_img.shape[2])
            
            # Replace the patch with zeros (black)
            occluded_img[0, i:i_end, j:j_end, :] = 0
            
            # Get prediction with the occluded image
            occluded_preds = model.predict(occluded_img, verbose=0)
            
            # Handle dictionary output
            if isinstance(occluded_preds, dict):
                occluded_pred_values = occluded_preds[pred_key]
            else:
                occluded_pred_values = occluded_preds
            
            # Get the new score for the original predicted class
            occluded_score = float(occluded_pred_values[0, pred_class])
            
            # Difference shows the importance of this region
            # The larger the drop, the more important this region is
            importance = pred_score - occluded_score
            
            # Assign to the importance map
            occlusion_map[i:i_end, j:j_end] = importance
            
            # Update progress
            processed_patches += 1
            if processed_patches % 50 == 0 or processed_patches == total_patches:
                print(f"Processed {processed_patches}/{total_patches} patches ({processed_patches/total_patches*100:.1f}%)")
    
    # Normalize the map to [0, 1]
    occlusion_map = np.maximum(occlusion_map, 0)  # Keep only positive values (where occlusion reduced the score)
    if np.max(occlusion_map) > 0:
        occlusion_map = occlusion_map / np.max(occlusion_map)
    
    return occlusion_map, pred_class, pred_score

def save_and_display_heatmap(img_path, heatmap, output_dir, class_names=None, pred_class=None, 
                            pred_score=None, alpha=0.7, colormap=cv2.COLORMAP_JET, 
                            label="Last Layer"):
    """
    Save and display visualization with enhanced contrast
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the heatmap to match the input image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply contrast enhancement to the heatmap
    # Convert the heatmap to RGB with higher intensity
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    
    # Create a slightly darkened version of the original image to make heatmap more visible
    darkened_img = np.clip(img * 0.7, 0, 255).astype('uint8')
    
    # Superimpose the heatmap on original image with higher alpha for more emphasis
    superimposed_img = heatmap_colored * alpha + darkened_img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save both the heatmap alone and the superimposed image
    image_filename = os.path.basename(img_path)
    base_filename = os.path.splitext(image_filename)[0]
    
    # Save the superimposed image
    overlay_path = os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_{base_filename}.jpg")
    cv2.imwrite(overlay_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    # Save the heatmap alone
    heatmap_path = os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_heatmap_{base_filename}.jpg")
    cv2.imwrite(heatmap_path, heatmap_colored)
    
    # Create title with prediction class and score if available
    title = label
    if class_names is not None and pred_class is not None and pred_score is not None:
        class_name = class_names[pred_class]
        title = f"{label}: {class_name} ({pred_score:.2f})"
    
    # Display the original image, the heatmap, and the superimposed image
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display the heatmap alone
    ax2.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    ax2.set_title('Heatmap')
    ax2.axis('off')
    
    ax3.imshow(superimposed_img)
    ax3.set_title(title)
    ax3.axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_comparison_{base_filename}.jpg")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{label} visualization saved to {overlay_path}")
    print(f"{label} heatmap saved to {heatmap_path}")
    print(f"Comparison image saved to {comparison_path}")
    
    return overlay_path, heatmap_path, comparison_path

def inspect_model_last_layer(model_path, img_path, output_dir, class_names=None, threshold=0.2, alpha=0.7):
    """
    Comprehensive inspection of the model's last layer
    """
    print("=== Starting Last Layer Inspection ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the model
    model = load_model_with_tfsm(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # 2. Print model summary
    print("\nModel Summary:")
    model.summary()
    
    print("\nAnalyzing last layer behavior...")
    
    # 3. Try GradCAM approach first
    print("Attempting GradCAM visualization...")
    original_img, preprocessed_img = preprocess_image(img_path)
    
    try:
        # Get heatmap using GradCAM
        heatmap, pred_class, pred_score = make_gradcam_heatmap_for_last_layer(
            preprocessed_img, model, threshold=threshold)
        
        if heatmap is not None:
            print("GradCAM visualization successful!")
            # Save and display GradCAM results
            save_and_display_heatmap(
                img_path, heatmap, output_dir, class_names, 
                pred_class, pred_score, alpha, cv2.COLORMAP_JET, "GradCAM Last Layer")
            
            visualization_method = "GradCAM"
            last_layer_heatmap = heatmap
        else:
            print("GradCAM failed. Trying occlusion sensitivity...")
            raise Exception("GradCAM returned None")
            
    except Exception as e:
        print(f"GradCAM failed: {e}")
        # 4. Try occlusion sensitivity as backup
        print("Using occlusion sensitivity for last layer visualization...")
        
        occlusion_map, pred_class, pred_score = visualize_last_layer_with_occlusion(
            model, img_path, output_dir, class_names)
        
        # Save and display occlusion results
        save_and_display_heatmap(
            img_path, occlusion_map, output_dir, class_names, 
            pred_class, pred_score, alpha, cv2.COLORMAP_JET, "Occlusion Last Layer")
        
        visualization_method = "Occlusion"
        last_layer_heatmap = occlusion_map
    
    # 5. Create a prediction summary
    print("\nPrediction Result:")
    if class_names and pred_class < len(class_names):
        print(f"Predicted class: {class_names[pred_class]} (Class index: {pred_class})")
    else:
        print(f"Predicted class index: {pred_class}")
    print(f"Confidence: {pred_score:.4f}")
    print(f"Visualization Method: {visualization_method}")
    
    print("=== Last Layer Inspection Complete ===")
    
    return model, last_layer_heatmap, pred_class, pred_score

# Example usage with your paths
if __name__ == "__main__":
    # Cassava disease class names
    class_names = [
        "Cassava Bacterial Blight (CBB)",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mite (CGM)",
        "Cassava Mosaic Disease (CMD)",
        "Healthy"
    ]
    
    # Your specific paths from the original file
    image_path = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\cassava-leaf-disease-classification\train_images\1919914.jpg"
    model_path = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\models"
    output_dir = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\images"
    
    # Visualization parameters
    threshold = 0.1  # Higher values will filter out more of the low-intensity areas
    alpha = 0.7      # Higher values will make the heatmap more prominent
    
    # Run the last layer inspection
    inspect_model_last_layer(model_path, image_path, output_dir, class_names, threshold, alpha)