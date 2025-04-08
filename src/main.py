import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.utils import plot_model

# Import the model functions from your module
# Assuming your code is in a file called cassava_classifier.py
from cassava_classifier import (
    load_folds, check_class_distribution, create_model, 
    train_kfold, train_final_model, predict_with_tta,
    NUM_CLASSES, DATASET_PATH, EPOCHS, BATCH_SIZE
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cassava Leaf Disease Classification')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                      help='Operation mode: train, predict, or evaluate')
    
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                      help='Path to the dataset folder containing fold CSV files')
    
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to a saved model file for prediction or evaluation')
    
    parser.add_argument('--image_path', type=str, default=None,
                      help='Path to an image file for prediction')
    
    parser.add_argument('--cross_validation', action='store_true',
                      help='Perform cross-validation training')
    
    parser.add_argument('--final_model', action='store_true',
                      help='Train a final model on all available data')
    
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save outputs (models, plots, etc.)')
    
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                      help='Number of training epochs')
    
    return parser.parse_args()

def setup_output_directory(base_dir):
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    return output_dir

def plot_training_history(history, output_path):
    """Plot training history metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train(args, output_dir):
    """Train models based on provided arguments"""
    print(f"Loading data from {args.dataset_path}")
    
    # Load dataset folds
    fold_data = load_folds(args.dataset_path)
    
    # Check if we have data
    if not fold_data:
        raise ValueError("No fold data was loaded. Check your file paths and CSV structure.")
    
    # Check class distribution
    correct_num_classes, unique_labels = check_class_distribution(fold_data)
    
    # Create log file
    log_file = os.path.join(output_dir, 'logs', 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset path: {args.dataset_path}\n")
        f.write(f"Number of classes: {NUM_CLASSES}\n")
        f.write(f"Found {correct_num_classes} unique classes in data\n")
        f.write(f"Unique labels: {unique_labels}\n\n")
    
    if args.cross_validation:
        print("Running K-fold cross-validation...")
        
        # Run K-fold cross-validation
        fold_models, fold_histories, fold_accuracies = train_kfold(fold_data)
        
        # Save cross-validation results
        cv_results = pd.DataFrame({
            'Fold': range(1, len(fold_accuracies) + 1),
            'Accuracy': fold_accuracies
        })
        cv_results.loc[len(cv_results)] = ['Mean', np.mean(fold_accuracies)]
        cv_results.loc[len(cv_results)] = ['Std Dev', np.std(fold_accuracies)]
        
        cv_results.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
        
        # Save each fold model
        for i, model in enumerate(fold_models):
            model_path = os.path.join(output_dir, 'models', f'fold_{i+1}_model.h5')
            model.save(model_path)
        
        # Plot histories for each fold
        for i, history_tuple in enumerate(fold_histories):
            if history_tuple[0] is not None:
                plot_path = os.path.join(output_dir, 'plots', f'fold_{i+1}_phase1_history.png')
                plot_training_history(history_tuple[0], plot_path)
            
            if history_tuple[1] is not None:
                plot_path = os.path.join(output_dir, 'plots', f'fold_{i+1}_phase2_history.png')
                plot_training_history(history_tuple[1], plot_path)
    
    if args.final_model:
        print("Training final model on all data...")
        
        # Combine all training data
        all_train_data = pd.concat([data['train'] for data in fold_data.values()])
        
        # Train final model
        final_model, final_history = train_final_model(all_train_data)
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'models', 'final_model.h5')
        final_model.save(final_model_path)
        
        # Plot final model history
        if final_history[0] is not None:
            plot_path = os.path.join(output_dir, 'plots', 'final_model_phase1_history.png')
            plot_training_history(final_history[0], plot_path)
        
        if final_history[1] is not None:
            plot_path = os.path.join(output_dir, 'plots', 'final_model_phase2_history.png')
            plot_training_history(final_history[1], plot_path)
        
        # Save model architecture diagram
        try:
            plot_model(final_model, 
                      to_file=os.path.join(output_dir, 'plots', 'model_architecture.png'),
                      show_shapes=True, 
                      show_layer_names=True)
        except Exception as e:
            print(f"Could not generate model architecture diagram: {e}")
        
        print(f"Final model saved to {final_model_path}")

def predict(args):
    """Make predictions on a single image"""
    if args.model_path is None or args.image_path is None:
        raise ValueError("Both --model_path and --image_path are required for prediction mode")
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False  # We don't need to load the custom loss function for inference
    )
    
    # Make prediction with TTA
    predicted_class, confidence, all_probabilities = predict_with_tta(model, args.image_path)
    
    # Print results
    print(f"\nPrediction for {os.path.basename(args.image_path)}:")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\nClass probabilities:")
    for i, prob in enumerate(all_probabilities):
        print(f"  Class {i}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Display the image
    try:
        import cv2
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Predicted: Class {predicted_class} ({confidence*100:.2f}%)")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image: {e}")

def evaluate(args):
    """Evaluate model on a test set"""
    if args.model_path is None:
        raise ValueError("--model_path is required for evaluation mode")
    
    # TODO: Implement evaluation function
    # This would load a test dataset and run evaluation metrics
    print("Evaluation mode not implemented yet")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory for training
    if args.mode == 'train':
        output_dir = setup_output_directory(args.output_dir)
        print(f"Output directory created at: {output_dir}")
    else:
        output_dir = None
    
    # Run the appropriate mode
    if args.mode == 'train':
        train(args, output_dir)
    elif args.mode == 'predict':
        predict(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Handle TensorFlow memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    
    # Run main function
    main()