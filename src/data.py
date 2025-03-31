import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from collections import Counter
import random
import cv2
from matplotlib.pyplot import figure

# Force matplotlib to use a different backend to avoid the Qt font error
matplotlib.use('Agg')

# Set the base directory path
base_dir = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\cassava-leaf-disease-classification"

# Define a custom output directory (change this path to your desired location)
images_dir = r"C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
print(f"Images will be saved to: {images_dir}")

# Load the label mapping
json_path = os.path.join(base_dir, 'label_num_to_disease_map.json')
with open(json_path, 'r') as f:
    label_map = json.load(f)

# Create abbreviated label map
abbreviated_label_map = {}
for key, value in label_map.items():
    if "Cassava Mosaic Disease" in value:
        abbreviated_label_map[key] = "CMD"
    elif "Cassava Green Mottle" in value:
        abbreviated_label_map[key] = "CGM"
    elif "Cassava Bacterial Blight" in value:
        abbreviated_label_map[key] = "CBB"
    elif "Cassava Brown Streak Disease" in value:
        abbreviated_label_map[key] = "CBSD"
    elif "Healthy" in value:
        abbreviated_label_map[key] = "Healthy"
    else:
        abbreviated_label_map[key] = value

# Load the training data
train_csv_path = os.path.join(base_dir, 'train.csv')
train_df = pd.read_csv(train_csv_path)

# Basic Dataset Information
print("=" * 50)
print("CASSAVA LEAF DISEASE CLASSIFICATION - DATA ANALYSIS")
print("=" * 50)
print(f"Total training samples: {len(train_df)}")

# Label distribution
class_counts = train_df['label'].value_counts().to_dict()
print("\nLabel distribution:")
for label_id, count in class_counts.items():
    disease_name = label_map.get(str(label_id), f"Unknown ({label_id})")
    abbr_name = abbreviated_label_map.get(str(label_id), f"Unknown ({label_id})")
    print(f"  {abbr_name} ({disease_name}): {count} images ({count/len(train_df)*100:.2f}%)")

# Plot class distribution with abbreviated labels
plt.figure(figsize=(10, 6))
plt.bar([abbreviated_label_map.get(str(id), f"Class {id}") for id in class_counts.keys()], 
        list(class_counts.values()))

# Set horizontal labels (flat, not inclined)
plt.xticks(rotation=0)

# Add more space at the bottom to accommodate the labels
plt.subplots_adjust(bottom=0.2)

# Add title and labels
plt.title('Distribution of Disease Classes')
plt.ylabel('Number of Images')
plt.xlabel('Disease Categories')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save figure to images folder
output_path = os.path.join(images_dir, 'disease_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.close()

# Check image properties and calculate RGB means by class
train_images_dir = os.path.join(base_dir, 'train_images')
if os.path.exists(train_images_dir):
    all_image_files = os.listdir(train_images_dir)
    print(f"\nTotal images in directory: {len(all_image_files)}")
    
    # Select 5 random images to display stats
    sample_images = random.sample(all_image_files, min(5, len(all_image_files)))
    
    print("\nSample Image Properties:")
    image_sizes = []
    for img_file in sample_images:
        img_path = os.path.join(train_images_dir, img_file)
        try:
            img = Image.open(img_path)
            print(f"Image {img_file}: Size={img.size}, Mode={img.mode}")
            image_sizes.append(img.size)
        except Exception as e:
            print(f"Error opening {img_file}: {e}")
    
    # Check if all images have the same dimensions
    if len(set(image_sizes)) == 1:
        print(f"\nAll sampled images have consistent dimensions: {image_sizes[0]}")
    else:
        print("\nImages have varying dimensions")
else:
    print(f"\nImage directory not found: {train_images_dir}")

# Calculate RGB mean values for each class
print("\nCalculating RGB mean values for each class...")
rgb_means = {label_id: {'R': [], 'G': [], 'B': []} for label_id in class_counts.keys()}
sample_size_per_class = min(100, min(class_counts.values()))  # Sample up to 100 images per class

for label_id in class_counts.keys():
    # Get file_names for this class
    class_files = train_df[train_df['label'] == label_id]['image_id'].tolist()
    
    # Select random samples
    selected_samples = random.sample(class_files, min(sample_size_per_class, len(class_files)))
    
    for img_id in selected_samples:
        img_path = os.path.join(train_images_dir, img_id)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Calculate mean per channel
                b, g, r = cv2.split(img)  # OpenCV uses BGR
                rgb_means[label_id]['R'].append(np.mean(r))
                rgb_means[label_id]['G'].append(np.mean(g))
                rgb_means[label_id]['B'].append(np.mean(b))

# Calculate the average RGB values for each class
rgb_averages = {}
for label_id, values in rgb_means.items():
    rgb_averages[label_id] = {
        'R': np.mean(values['R']) if values['R'] else 0,
        'G': np.mean(values['G']) if values['G'] else 0,
        'B': np.mean(values['B']) if values['B'] else 0
    }

# Display RGB averages
print("\nRGB Mean Values by Class:")
for label_id, values in rgb_averages.items():
    abbr_name = abbreviated_label_map.get(str(label_id), f"Class {label_id}")
    print(f"  {abbr_name}: R={values['R']:.2f}, G={values['G']:.2f}, B={values['B']:.2f}")

# Plot RGB means by class
labels = [abbreviated_label_map.get(str(id), f"Class {id}") for id in rgb_averages.keys()]
r_means = [rgb_averages[id]['R'] for id in rgb_averages.keys()]
g_means = [rgb_averages[id]['G'] for id in rgb_averages.keys()]
b_means = [rgb_averages[id]['B'] for id in rgb_averages.keys()]

plt.figure(figsize=(10, 6))
bar_width = 0.25
r_bars = np.arange(len(labels))
g_bars = [x + bar_width for x in r_bars]
b_bars = [x + bar_width for x in g_bars]

plt.bar(r_bars, r_means, width=bar_width, label='Red', color='red', alpha=0.7)
plt.bar(g_bars, g_means, width=bar_width, label='Green', color='green', alpha=0.7)
plt.bar(b_bars, b_means, width=bar_width, label='Blue', color='blue', alpha=0.7)

plt.xlabel('Disease Class')
plt.ylabel('Mean Pixel Value')
plt.title('Mean RGB Values by Disease Class')
plt.xticks([r + bar_width for r in range(len(labels))], labels)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'rgb_means_by_class.png'), dpi=300)
plt.close()

# Display single sample image from each disease class (excluding Healthy)
plt.figure(figsize=(15, 4))

# Load a single sample image from each disease class
class_examples = {}
class_names = {}
disease_labels = []

# Identify disease class labels (exclude Healthy)
for label_id in class_counts.keys():
    abbr_name = abbreviated_label_map.get(str(label_id), f"Class {label_id}")
    if abbr_name != "Healthy":
        disease_labels.append(label_id)
        class_names[label_id] = abbr_name

# Get class files
for label_id in disease_labels:
    # Get file_names for this class
    class_files = train_df[train_df['label'] == label_id]['image_id'].tolist()
    
    # Select random sample
    if class_files:
        img_id = random.choice(class_files)
        img_path = os.path.join(train_images_dir, img_id)
        if os.path.exists(img_path):
            class_examples[label_id] = img_path

# Plot examples in a tight grid
num_classes = len(disease_labels)
fig, axes = plt.subplots(1, num_classes, figsize=(15, 4))
fig.suptitle('Sample Image from Each Disease Class', fontsize=16)

# Populate the grid
for i, label_id in enumerate(disease_labels):
    if label_id in class_examples:
        img_path = class_examples[label_id]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Getting central crop for closer view
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_size = min(w, h) // 2  # Get a square crop of half the smallest dimension
        crop = img[center_y-crop_size:center_y+crop_size, center_x-crop_size:center_x+crop_size]
        
        axes[i].imshow(crop)
        axes[i].set_title(class_names[label_id], fontsize=12)
        axes[i].axis('off')

# Make layout tight with minimal white space
plt.tight_layout()
plt.subplots_adjust(top=0.85, wspace=0.1)  # Reduce white space between plots
plt.savefig(os.path.join(images_dir, 'disease_class_examples.png'), dpi=300, bbox_inches='tight')
plt.close()

# Calculate image statistics
print("\nCalculating image statistics...")
brightness_values = []
contrast_values = []
if os.path.exists(train_images_dir) and len(all_image_files) > 0:
    # Filter for valid image files
    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in all_image_files if os.path.splitext(f.lower())[1] in valid_image_extensions]
    
    if len(image_files) > 0:
        sample_size = min(500, len(image_files))  # Sample a subset for speed
        random_images = random.sample(image_files, sample_size)
    else:
        random_images = []
        print("No valid image files found for statistics calculation.")
else:
    random_images = []
    print("No images directory or empty directory for statistics calculation.")

for img_file in random_images:
    img_path = os.path.join(train_images_dir, img_file)
    try:
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for statistics
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        brightness_values.append(brightness)
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        contrast_values.append(contrast)
    except:
        continue

# Plot brightness and contrast distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(brightness_values, bins=30, color='skyblue', edgecolor='black')
plt.title('Brightness Distribution')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Number of Images')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(contrast_values, bins=30, color='lightgreen', edgecolor='black')
plt.title('Contrast Distribution')
plt.xlabel('Standard Deviation of Pixel Values')
plt.ylabel('Number of Images')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'image_statistics.png'), dpi=300)
plt.close()

# NEW CODE: Image enhancement demonstration
print("\n" + "=" * 50)
print("IMAGE ENHANCEMENT DEMONSTRATION")
print("=" * 50)

# Select a random image for enhancement demonstration
if os.path.exists(train_images_dir) and len(all_image_files) > 0:
    # Filter for only image files
    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in all_image_files if os.path.splitext(f.lower())[1] in valid_image_extensions]
    
    if len(image_files) > 0:
        demo_image_file = random.choice(image_files)
        demo_image_path = os.path.join(train_images_dir, demo_image_file)
        print(f"Selected image for enhancement demonstration: {demo_image_file}")
    else:
        print("No valid image files found in directory.")
        demo_image_path = None
    
    # Read the image
    original_img = cv2.imread(demo_image_path)
    if original_img is not None:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        print(f"Successfully loaded image for enhancement: {demo_image_file}")
        
        # Create enhanced versions
        h, w = original_img.shape[:2]
        
        # 1. Horizontal Flip
        h_flip_img = cv2.flip(original_img, 1)  # 1 for horizontal flip
        
        # 2. Vertical Flip
        v_flip_img = cv2.flip(original_img, 0)  # 0 for vertical flip
        
        # 3. Rotation (45 degrees)
        rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
        rotated_img = cv2.warpAffine(original_img, rotation_matrix, (w, h))
        
        # 4. Center Crop
        center_x, center_y = w // 2, h // 2
        crop_size = min(w, h) // 2
        center_crop_img = original_img[center_y-crop_size:center_y+crop_size, center_x-crop_size:center_x+crop_size]
        
        # 5. Random Crop
        offset_x = random.randint(0, max(0, w - crop_size))
        offset_y = random.randint(0, max(0, h - crop_size))
        random_crop_img = original_img[offset_y:offset_y+crop_size, offset_x:offset_x+crop_size]
        
        # 6. Brightness Adjustment
        # Increase brightness
        bright_img = cv2.convertScaleAbs(original_img, alpha=1.5, beta=30)
        
        # 7. Contrast Enhancement (Low)
        low_contrast_img = cv2.convertScaleAbs(original_img, alpha=0.7, beta=0)
        
        # 8. Contrast Enhancement (High)
        high_contrast_img = cv2.convertScaleAbs(original_img, alpha=1.8, beta=0)
        
        # 9. CLAHE Contrast Enhancement
        lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        clahe_img = cv2.merge((cl, a, b))
        clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)
        
        # 10. Hue Adjustment
        hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + 20) % 180  # Shift hue
        hsv = hsv.astype(np.uint8)
        hue_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 11. Saturation Adjustment (Increase)
        hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * 1.5  # Increase saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        hsv = hsv.astype(np.uint8)
        high_sat_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 12. Saturation Adjustment (Decrease)
        hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,1] = hsv[:,:,1] * 0.5  # Decrease saturation
        hsv = hsv.astype(np.uint8)
        low_sat_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 13. JPEG Quality Simulation
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]  # Low quality JPEG
        _, encoded_img = cv2.imencode('.jpg', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), encode_param)
        low_quality_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        low_quality_img = cv2.cvtColor(low_quality_img, cv2.COLOR_BGR2RGB)
        
        # Create figures to display all enhancements in multiple panels
        # First group - Geometric transformations
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
        fig1.suptitle('Geometric Transformations for Data Augmentation', fontsize=16)
        
        # Display original and geometric transformations
        axes1[0, 0].imshow(original_img)
        axes1[0, 0].set_title('Original Image')
        axes1[0, 0].axis('off')
        
        axes1[0, 1].imshow(h_flip_img)
        axes1[0, 1].set_title('Horizontal Flip')
        axes1[0, 1].axis('off')
        
        axes1[0, 2].imshow(v_flip_img)
        axes1[0, 2].set_title('Vertical Flip')
        axes1[0, 2].axis('off')
        
        axes1[1, 0].imshow(rotated_img)
        axes1[1, 0].set_title('45° Rotation')
        axes1[1, 0].axis('off')
        
        axes1[1, 1].imshow(center_crop_img)
        axes1[1, 1].set_title('Center Crop')
        axes1[1, 1].axis('off')
        
        axes1[1, 2].imshow(random_crop_img)
        axes1[1, 2].set_title('Random Crop')
        axes1[1, 2].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for title
        geo_transform_path = os.path.join(images_dir, 'geometric_transformations.png')
        plt.savefig(geo_transform_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Second group - Color and quality adjustments
        fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
        fig2.suptitle('Color and Quality Adjustments for Data Augmentation', fontsize=16)
        
        # Display color transformations
        axes2[0, 0].imshow(original_img)
        axes2[0, 0].set_title('Original Image')
        axes2[0, 0].axis('off')
        
        axes2[0, 1].imshow(bright_img)
        axes2[0, 1].set_title('Brightness Increase')
        axes2[0, 1].axis('off')
        
        axes2[0, 2].imshow(low_contrast_img)
        axes2[0, 2].set_title('Low Contrast')
        axes2[0, 2].axis('off')
        
        axes2[1, 0].imshow(high_contrast_img)
        axes2[1, 0].set_title('High Contrast')
        axes2[1, 0].axis('off')
        
        axes2[1, 1].imshow(clahe_img)
        axes2[1, 1].set_title('CLAHE Enhancement')
        axes2[1, 1].axis('off')
        
        axes2[1, 2].imshow(hue_img)
        axes2[1, 2].set_title('Hue Shift')
        axes2[1, 2].axis('off')
        
        axes2[2, 0].imshow(high_sat_img)
        axes2[2, 0].set_title('Increased Saturation')
        axes2[2, 0].axis('off')
        
        axes2[2, 1].imshow(low_sat_img)
        axes2[2, 1].set_title('Decreased Saturation')
        axes2[2, 1].axis('off')
        
        axes2[2, 2].imshow(low_quality_img)
        axes2[2, 2].set_title('JPEG Quality Reduction')
        axes2[2, 2].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for title
        
        # Save the color enhancement demonstration
        color_enhancement_path = os.path.join(images_dir, 'color_quality_adjustments.png')
        plt.savefig(color_enhancement_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a comprehensive figure showing all transformations
        # This will be a larger grid with all 13 transformations plus original
        fig3, axes3 = plt.subplots(4, 4, figsize=(20, 20))
        fig3.suptitle('Comprehensive Data Augmentation Techniques', fontsize=18)
        
        # Flatten the axes for easier indexing
        axes3 = axes3.flatten()
        
        # List of all images and their titles
        all_images = [
            (original_img, 'Original Image'),
            (h_flip_img, 'Horizontal Flip'),
            (v_flip_img, 'Vertical Flip'),
            (rotated_img, '45° Rotation'),
            (center_crop_img, 'Center Crop'),
            (random_crop_img, 'Random Crop'),
            (bright_img, 'Brightness Increase'),
            (low_contrast_img, 'Low Contrast'),
            (high_contrast_img, 'High Contrast'),
            (clahe_img, 'CLAHE Enhancement'),
            (hue_img, 'Hue Shift'),
            (high_sat_img, 'Increased Saturation'),
            (low_sat_img, 'Decreased Saturation'),
            (low_quality_img, 'JPEG Quality Reduction')
        ]
        
        # Plot each image
        for i, (img, title) in enumerate(all_images):
            axes3[i].imshow(img)
            axes3[i].set_title(title)
            axes3[i].axis('off')
        
        # Turn off any unused axes
        for i in range(len(all_images), len(axes3)):
            axes3[i].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust for title
        
        # Save the comprehensive demonstration
        all_enhancement_path = os.path.join(images_dir, 'all_augmentation_techniques.png')
        plt.savefig(all_enhancement_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # NEW: Create a visualization showing requested augmentations on the same input image
        fig4, axes4 = plt.subplots(3, 3, figsize=(15, 15))
        fig4.suptitle('Selected Augmentations on Same Input Image', fontsize=18)
        
        # Display the selected augmentations
        axes4[0, 0].imshow(original_img)
        axes4[0, 0].set_title('Original Image')
        axes4[0, 0].axis('off')
        
        axes4[0, 1].imshow(center_crop_img)
        axes4[0, 1].set_title('Center Crop')
        axes4[0, 1].axis('off')
        
        axes4[0, 2].imshow(low_contrast_img)
        axes4[0, 2].set_title('Low Contrast')
        axes4[0, 2].axis('off')
        
        axes4[1, 0].imshow(hue_img)
        axes4[1, 0].set_title('Hue Shift')
        axes4[1, 0].axis('off')
        
        axes4[1, 1].imshow(clahe_img)
        axes4[1, 1].set_title('CLAHE Enhancement')
        axes4[1, 1].axis('off')
        
        axes4[1, 2].imshow(high_sat_img)
        axes4[1, 2].set_title('Increased Saturation')
        axes4[1, 2].axis('off')
        
        axes4[2, 0].imshow(low_quality_img)
        axes4[2, 0].set_title('JPEG Quality Reduction')
        axes4[2, 0].axis('off')
        
        axes4[2, 1].imshow(high_contrast_img)
        axes4[2, 1].set_title('High Contrast')
        axes4[2, 1].axis('off')
        
        axes4[2, 2].imshow(bright_img)
        axes4[2, 2].set_title('Brightness Increase')
        axes4[2, 2].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust for title
        
        # Save the selected augmentations visualization
        selected_augmentations_path = os.path.join(images_dir, 'selected_augmentations.png')
        plt.savefig(selected_augmentations_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Verify the files were saved
        saved_paths = [geo_transform_path, color_enhancement_path, all_enhancement_path, selected_augmentations_path]
        for path in saved_paths:
            if os.path.exists(path):
                print(f"Successfully saved: {path}")
            else:
                print(f"Failed to save: {path}")
        
        print(f"Image enhancement demonstration saved to {os.path.join(images_dir, 'enhancement_demonstration.png')}")
    else:
        print(f"Failed to load the demonstration image: {demo_image_path}")
        print("Error details: Image could not be read or is None")
else:
    print("No images available for enhancement demonstration. Check if the train_images_dir path is correct.")

# Summarize findings
print("\n" + "=" * 50)
print("DATA ANALYSIS SUMMARY")
print("=" * 50)
print(f"1. Dataset contains {len(train_df)} labeled images across {len(class_counts)} classes")
print("2. Class distribution is imbalanced:")
for label_id, count in class_counts.items():
    abbr_name = abbreviated_label_map.get(str(label_id), f"Unknown ({label_id})")
    print(f"   - {abbr_name}: {count} images ({count/len(train_df)*100:.2f}%)")

print("3. Images appear to have consistent dimensions")
print(f"4. Average brightness: {np.mean(brightness_values):.2f}")
print(f"5. Average contrast: {np.mean(contrast_values):.2f}")

print("\n6. RGB mean values by class:")
for label_id, values in rgb_averages.items():
    abbr_name = abbreviated_label_map.get(str(label_id), f"Class {label_id}")
    print(f"   - {abbr_name}: R={values['R']:.2f}, G={values['G']:.2f}, B={values['B']:.2f}")

print("\n7. Image Enhancement Evaluation:")
print("   - Horizontal Flip: Reasonable - maintains leaf structure while increasing data diversity")
print("   - Vertical Flip: Reasonable - although less common in natural settings, adds variation")
print("   - Rotation: Reasonable - leaves can be viewed from different angles in natural settings")
print("   - Center Crop: Reasonable - focuses on specific leaf regions with disease symptoms")
print("   - Random Crop: Reasonable - simulates different viewing frames and partial leaf views")
print("   - Brightness Adjustment: Reasonable - simulates different lighting conditions")
print("   - Low Contrast: Reasonable - simulates cloudy conditions or shade")
print("   - High Contrast: Reasonable - can help highlight disease features")
print("   - CLAHE Enhancement: Reasonable - enhances local contrast to highlight details")
print("   - Hue Shift: Use with caution - may alter disease appearance, but helps model robustness")
print("   - Increased Saturation: Use with caution - intensifies colors but may exaggerate symptoms")
print("   - Decreased Saturation: Reasonable - simulates older leaves or faded conditions")
print("   - JPEG Quality Reduction: Reasonable - helps model be robust to image quality variations")

print("\nRecommendations for model development:")
print("1. Address class imbalance using techniques like:")
print("   - Class weights in loss function")
print("   - Oversampling minority classes")
print("   - Data augmentation (as demonstrated)")
print("2. Consider image preprocessing to normalize brightness/contrast")
print("3. Use transfer learning with pre-trained CNNs like ResNet, EfficientNet")
print("4. Implement cross-validation to ensure robust performance across all classes")
print("5. Monitor per-class metrics (not just overall accuracy)")
print("6. The differences in RGB means between classes might be useful features for classification")
print("7. Apply the demonstrated augmentation techniques to increase dataset size and diversity")

print("\nAnalysis complete! Visualizations saved to:")
print(f"- {os.path.join(images_dir, 'disease_distribution.png')}")
print(f"- {os.path.join(images_dir, 'disease_class_examples.png')} (CBSD, CBB, CMD, CGM only)")
print(f"- {os.path.join(images_dir, 'image_statistics.png')}")
print(f"- {os.path.join(images_dir, 'rgb_means_by_class.png')}")
print(f"- {os.path.join(images_dir, 'geometric_transformations.png')}")
print(f"- {os.path.join(images_dir, 'color_quality_adjustments.png')}")
print(f"- {os.path.join(images_dir, 'all_augmentation_techniques.png')}")
print(f"- {os.path.join(images_dir, 'selected_augmentations.png')} (Requested specific augmentations)")