import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
import argparse
from tqdm import tqdm
import glob
import json
import csv

def create_folds(
    data_path, 
    num_folds=5,
    stratify_col='label',
    group_col=None,
    seed=42, 
    output_dir='folds'
):
    """
    Create stratified k-folds for a dataset and save the fold assignments.
    
    Args:
        data_path: Path to CSV file with dataset info or directory of images
        num_folds: Number of folds to create
        stratify_col: Column name to use for stratification (usually class labels)
        group_col: Column name to use for grouped k-fold (if needed)
        seed: Random seed for reproducibility
        output_dir: Directory to save fold information
    
    Returns:
        DataFrame with original data plus fold assignments
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or create dataset DataFrame
    if data_path.endswith('.csv'):
        print(f"Loading dataset from CSV: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"Creating dataset from directory: {data_path}")
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
        
        # Extract class from directory structure (assuming class is parent folder name)
        data = []
        for img_path in tqdm(image_files, desc="Processing images"):
            # Get relative path
            rel_path = os.path.relpath(img_path, data_path)
            # For typical structure: class_name/image.jpg
            class_name = os.path.dirname(rel_path)
            # If using nested directories, you might need to adjust this
            if class_name == '':
                class_name = 'unknown'  # Handle flat directory structure
            
            data.append({
                'image_path': img_path,
                'class_name': class_name
            })
        
        df = pd.DataFrame(data)
        
        # Convert class names to numeric labels if needed
        if 'label' not in df.columns and 'class_name' in df.columns:
            # Create a mapping from class names to integer labels
            class_mapping = {name: idx for idx, name in enumerate(df['class_name'].unique())}
            df['label'] = df['class_name'].map(class_mapping)
            
            # Save the class mapping for future reference
            with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
                json.dump(class_mapping, f, indent=2)
    
    # Verify the stratify column exists
    if stratify_col not in df.columns:
        raise ValueError(f"Stratification column '{stratify_col}' not found in the dataset. "
                         f"Available columns: {df.columns.tolist()}")
    
    # Create folds
    if group_col is not None and group_col in df.columns:
        print(f"Creating {num_folds} folds with GroupKFold using '{group_col}' column")
        kf = GroupKFold(n_splits=num_folds)
        fold_indices = list(kf.split(df, df[stratify_col], groups=df[group_col]))
    else:
        print(f"Creating {num_folds} stratified folds using '{stratify_col}' column")
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_indices = list(skf.split(df, df[stratify_col]))
    
    # Add fold column to the DataFrame
    df['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(fold_indices):
        df.loc[val_idx, 'fold'] = fold_idx
    
    # Analyze fold distribution
    print("\nClass distribution across folds:")
    fold_stats = df.groupby(['fold', stratify_col]).size().unstack().fillna(0)
    print(fold_stats)
    print("\nFold sizes:")
    print(df.groupby('fold').size())
    
    # Save the DataFrame with fold assignments
    output_path = os.path.join(output_dir, 'dataset_with_folds.csv')
    df.to_csv(output_path, index=False)
    print(f"\nFold assignments saved to: {output_path}")
    
    # Create individual fold files
    for fold in range(num_folds):
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        train_df.to_csv(os.path.join(output_dir, f'train_fold{fold}.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, f'val_fold{fold}.csv'), index=False)
    
    print(f"Individual fold files created in: {output_dir}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create K-folds for a dataset')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to dataset CSV or directory')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds to create')
    parser.add_argument('--stratify_col', type=str, default='label',
                        help='Column to use for stratification')
    parser.add_argument('--group_col', type=str, default=None,
                        help='Column to use for GroupKFold (if needed)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='folds',
                        help='Directory to save fold information')
    
    args = parser.parse_args()
    
    df_with_folds = create_folds(
        args.data_path,
        args.num_folds,
        args.stratify_col,
        args.group_col,
        args.seed,
        args.output_dir
    )