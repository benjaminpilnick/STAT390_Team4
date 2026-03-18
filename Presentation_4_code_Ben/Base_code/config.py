"""
Configuration file for MIL training
"""
import os
from typing import Tuple

# Data paths (adjust these for your HPC environment)
DATA_PATHS = {
    'labels_csv': '/projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/extra_benign_case_grade_match.csv',
    'patches_dir': '/projects/e32998/patches_benign_split',
    'runs_dir': '/projects/e32998/MIL_training/final_runs'  # Base directory for training runs
}

# Model configuration
MODEL_CONFIG = {
    'num_classes': 2,
    'embed_dim': 512,
    'attention_hidden_dim': 128,
    'per_slice_cap': 800,
    'max_slices_per_stain': None,
    'stains': ('h&e', 'melan', 'sox10')
}


#OPTUNA Hyperaparameters (from the best trial: 5 in the study. Validation loss of 0.25)
OPTUNA_HYPERPARAMS = {  
    'class_weight_benign': 2.13658618927002,
    'classifier_dropout': 0.198203802108765,
    'entropy_lambda': 0.000997635683829176,
    'learning_rate': 5.22465277102519e-05,
    'patch_proj_dropout': 0.511009573936462,
    'weight_decay': 0.000469622726832617
}

OPTUNA_HYPERPARAMS2 = {
    'class_weight_benign': 2.88660573959351,
    'classifier_dropout': 0.331241250038147,
    'entropy_lambda': 4.78111935799506e-05,
    'learning_rate': 6.09039212221789e-05,
    'patch_proj_dropout': 0.49042671918869,
    'weight_decay': 4.44088475879475e-05
}

OPTUNA_HYPERPARAMS3 = {
    'class_weight_benign': 2.88660573959351,
    'classifier_dropout': 0.331241250038147,
    'entropy_lambda': 2.83804641322124e-05,
    'learning_rate': 4.61454938727795e-05,
    'patch_proj_dropout': 0.5203657746315,
    'weight_decay': 2.2298951944635e-05
}


#Updating with the most recent Hyperparams from OPTUNA Study
# Training configuration
# Training configuration
TRAINING_CONFIG = {
    'epochs': 30,  # Increased since we have early stopping
    'batch_size': 1,  # MIL typically uses batch_size=1
    'learning_rate': OPTUNA_HYPERPARAMS3['learning_rate'],  # Updated from OPTUNA study
    'weight_decay': OPTUNA_HYPERPARAMS3['weight_decay'],  # Updated from OPTUNA study
    'num_workers': 2,
    'pin_memory': True,
    'random_state': 42,
    'class_weights': [OPTUNA_HYPERPARAMS3['class_weight_benign'], 1.0],  # Increased benign weight from 2.0 to 2.5285 (from OPTUNA study)
    'classifier_dropout': OPTUNA_HYPERPARAMS3['classifier_dropout'],  # Add dropout for regularization
    'patch_proj_dropout': OPTUNA_HYPERPARAMS3['patch_proj_dropout'],  # Add dropout for patch projection
    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',  # 'reduce_on_plateau' or 'cosine'
    'scheduler_patience': 4,  # For ReduceLROnPlateau
    'scheduler_factor': 0.2139,  # Reduce LR by half
    'scheduler_min_lr': 1e-6,
    # Early stopping
    'early_stopping': True,
    'early_stopping_patience': 8,  # Stop if no improvement for 10 epochs
    'early_stopping_min_delta': 0.001,  # Minimum change to qualify as improvement
    'early_stopping_min_epochs': 10,  # Minimum epochs before early stopping can trigger

    'use_patch_entropy_regularization': True,
    'patch_entropy_lambda': OPTUNA_HYPERPARAMS3['entropy_lambda'],  # Updated from OPTUNA study
    'patch_entropy_mode': "max",   # "max" or "min"
    'patch_entropy_eps': 1e-8,
    'normalize_patch_entropy': True
}


# Data split configuration
SPLIT_CONFIG = {
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'stratify': True
}

# Related benign cases
GROUPED_CASES = [(22, 107, 108), (24, 118), (25, 119), (26, 109, 110, 111)]

# Image preprocessing
IMAGE_CONFIG = {
    'image_size': (224, 224),
    'normalize_mean': [0.485, 0.456, 0.406],  # DenseNet mean
    'normalize_std': [0.229, 0.224, 0.225]   # DenseNet std
}

# Valid classes for filtering
VALID_CLASSES = [1.0, 3.0, 4.0]

# Device configuration
DEVICE = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'true').lower() == 'true' else 'cpu'
