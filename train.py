import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import KFold
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMAGE_SIZE = 512  # Using 512x512 as per specifications
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 5  # Update based on your dataset (assuming Cassava dataset)
DATASET_PATH = r'C:\Users\SIMON\Desktop\AMLS2_Cassava_CV\folds'
NUM_FOLDS = 5
DROP_CONNECT_RATE = 0.4
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25
LABEL_SMOOTHING = 0.1
MAX_LR = 0.0002
MIN_LR_START = 1e-6
MIN_LR_END = 3.17e-6
WARMUP_EPOCHS = 2

# Cassava dataset mean and std values
GLOBAL_MEAN = [0.4316205, 0.49817887, 0.31532103]
GLOBAL_STD = [0.21871215, 0.22380711, 0.20057836]

# Custom Sigmoid Focal Loss with Label Smoothing
class SigmoidFocalLossWithLabelSmoothing(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, from_logits=False, **kwargs):
        super(SigmoidFocalLossWithLabelSmoothing, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.bce = BinaryCrossentropy(
            from_logits=from_logits, 
            label_smoothing=label_smoothing, 
            reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Apply label smoothing
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        
        # Calculate binary cross entropy
        bce_loss = self.bce(y_true, y_pred)
        
        # Add focal loss factor
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Return focal loss
        return tf.reduce_mean(alpha_factor * modulating_factor * bce_loss)

# Custom learning rate scheduler with warmup and cosine decay
class WarmUpCosineDecayScheduler(Callback):
    def __init__(self, 
                 min_lr_start=1e-6,
                 max_lr=0.0002,
                 min_lr_end=3.17e-6,
                 warmup_epochs=2,
                 total_epochs=20):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.min_lr_start = min_lr_start
        self.max_lr = max_lr
        self.min_lr_end = min_lr_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr_start + (self.max_lr - self.min_lr_start) * (epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * progress))
            lr = self.min_lr_end + (self.max_lr - self.min_lr_end) * cosine_decay
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"\nEpoch {epoch+1}/{self.total_epochs}, LR: {lr:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

# Advanced augmentation functions using albumentations
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(p=0.5, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=0.5),
        A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
    ])

# Dataset class to handle custom transformations
class CassavaDataset:
    def __init__(self, image_paths, labels=None, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']
        
        # Normalize using global mean and std if not done in transforms
        img = img / 255.0
        
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        return img

# TTA (Test Time Augmentation) functions
def get_tta_transforms():
    tta_transforms = [
        # Original image (center crop)
        A.Compose([
            A.CenterCrop(height=512, width=512),
            A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
        ]),
        # Horizontal flip
        A.Compose([
            A.CenterCrop(height=512, width=512),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
        ]),
        # Vertical flip
        A.Compose([
            A.CenterCrop(height=512, width=512),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
        ]),
        # Transpose
        A.Compose([
            A.CenterCrop(height=512, width=512),
            A.Transpose(p=1.0),
            A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
        ]),
        # 90-degree rotation
        A.Compose([
            A.CenterCrop(height=512, width=512),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=GLOBAL_MEAN, std=GLOBAL_STD),
        ])
    ]
    return tta_transforms

def get_tta_patches(image, original_size=(800, 600)):
    h, w = original_size
    patch_size = 512
    
    # Define overlapping patches
    patches = [
        (0, 0, patch_size, patch_size),  # top-left
        (w - patch_size, 0, w, patch_size),  # top-right
        (0, h - patch_size, patch_size, h),  # bottom-left
        (w - patch_size, h - patch_size, w, h),  # bottom-right
        ((w - patch_size) // 2, (h - patch_size) // 2, 
         (w + patch_size) // 2, (h + patch_size) // 2)  # center
    ]
    
    patch_images = []
    for x1, y1, x2, y2 in patches:
        patch = image[y1:y2, x1:x2]
        patch_images.append(patch)
    
    return patch_images

def apply_tta(model, image_path, tta_transforms):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Get patches
    patches = get_tta_patches(img, original_size=(w, h))
    
    all_predictions = []
    
    # Apply augmentations to patches
    for patch in patches:
        # Apply all transforms to each patch
        for transform in tta_transforms:
            augmented = transform(image=patch)
            transformed_patch = augmented['image']
            
            # Convert to tensor and add batch dimension
            transformed_patch = np.expand_dims(transformed_patch, axis=0)
            
            # Make prediction
            pred = model.predict(transformed_patch)
            all_predictions.append(pred[0])
    
    # Average all predictions
    final_prediction = np.mean(all_predictions, axis=0)
    return final_prediction

# Model Creation with drop connect rate
def create_model():
    # Load pre-trained EfficientNetB4 with drop connect rate
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Apply custom normalization
    x = BatchNormalization()(inputs)
    
    # Load EfficientNetB4 with specific drop connect rate
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_tensor=x,
        drop_connect_rate=DROP_CONNECT_RATE
    )
    
    # Freeze the base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers as specified
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=predictions)
    
    # Compile with custom loss function
    model.compile(
        optimizer=Adam(learning_rate=MIN_LR_START),
        loss=SigmoidFocalLossWithLabelSmoothing(
            gamma=FOCAL_LOSS_GAMMA, 
            alpha=FOCAL_LOSS_ALPHA, 
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )
    
    return model, base_model

# K-Fold Cross Validation training
def train_kfold(df):
    # Setup KFold
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # Lists for tracking performance
    fold_accuracies = []
    fold_models = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"\n{'='*20} Fold {fold+1}/{NUM_FOLDS} {'='*20}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Create data generators with custom transformations
        train_dataset = CassavaDataset(
            image_paths=train_df['image_path'].values,
            labels=train_df['label'].values,
            transforms=get_train_transforms()
        )
        
        val_dataset = CassavaDataset(
            image_paths=val_df['image_path'].values,
            labels=val_df['label'].values,
            transforms=get_valid_transforms()
        )
        
        # Create model
        model, base_model = create_model()
        
        # Setup callbacks
        checkpoint = ModelCheckpoint(
            f'efficientnetb4_fold{fold+1}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        lr_scheduler = WarmUpCosineDecayScheduler(
            min_lr_start=MIN_LR_START,
            max_lr=MAX_LR,
            min_lr_end=MIN_LR_END,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=EPOCHS
        )
        
        callbacks = [checkpoint, early_stopping, lr_scheduler]
        
        # First phase: Train only the top layers
        print("Phase 1: Training only the top layers...")
        history_1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=5,
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )
        
        # Second phase: Fine-tune the model
        print("Phase 2: Fine-tuning the model...")
        
        # Unfreeze the model gradually
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        
        # Continue training
        history_2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            initial_epoch=5,
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(val_dataset, batch_size=BATCH_SIZE)
        print(f"Fold {fold+1} - Validation Accuracy: {val_acc:.4f}")
        
        fold_accuracies.append(val_acc)
        fold_models.append(model)
        fold_histories.append((history_1, history_2))
    
    # Print final cross-validation results
    print(f"\n{'='*20} K-Fold Cross-Validation Results {'='*20}")
    for fold, acc in enumerate(fold_accuracies):
        print(f"Fold {fold+1}: {acc:.4f}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Dev: {np.std(fold_accuracies):.4f}")
    
    return fold_models, fold_histories, fold_accuracies

# Train final model on entire dataset
def train_final_model(df):
    print(f"\n{'='*20} Training Final Model {'='*20}")
    
    # Create dataset with all data
    full_dataset = CassavaDataset(
        image_paths=df['image_path'].values,
        labels=df['label'].values,
        transforms=get_train_transforms()
    )
    
    # Create a small validation set (10%) for monitoring
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    val_dataset = CassavaDataset(
        image_paths=val_df['image_path'].values,
        labels=val_df['label'].values,
        transforms=get_valid_transforms()
    )
    
    # Create model
    model, base_model = create_model()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        'efficientnetb4_final_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    lr_scheduler = WarmUpCosineDecayScheduler(
        min_lr_start=MIN_LR_START,
        max_lr=MAX_LR,
        min_lr_end=MIN_LR_END,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=14  # Train for 14 epochs as specified
    )
    
    callbacks = [checkpoint, lr_scheduler]
    
    # First phase: Train only the top layers
    print("Phase 1: Training only the top layers...")
    history_1 = model.fit(
        full_dataset,
        validation_data=val_dataset,
        epochs=3,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Second phase: Fine-tune the model
    print("Phase 2: Fine-tuning the model...")
    
    # Unfreeze the model gradually
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Continue training for a total of 14 epochs as specified
    history_2 = model.fit(
        full_dataset,
        validation_data=val_dataset,
        epochs=14,
        initial_epoch=3,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save('efficientnetb4_final_model.h5')
    
    return model, (history_1, history_2)

# Function to make predictions with TTA
def predict_with_tta(model, image_path):
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    
    # Apply TTA and get final prediction
    prediction = apply_tta(model, image_path, tta_transforms)
    
    # Get predicted class
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    return predicted_class, confidence, prediction

# Main execution
if __name__ == "__main__":
    print("Starting EfficientNetB4 Transfer Learning with specified parameters...")
    
    # Prepare your dataset dataframe with image paths and labels
    # This is a placeholder - you should replace it with your actual data loading code
    df = pd.DataFrame({
        'image_path': ['path/to/image1.jpg', 'path/to/image2.jpg'],
        'label': [0, 1]
    })
    
    # Run K-fold cross-validation
    fold_models, fold_histories, fold_accuracies = train_kfold(df)
    
    # Train final model on entire dataset
    final_model, final_history = train_final_model(df)
    
    print("Training completed successfully!")
    
    # Example of making a prediction with TTA
    test_image_path = 'path/to/test_image.jpg'
    predicted_class, confidence, _ = predict_with_tta(final_model, test_image_path)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")