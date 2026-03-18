#!/usr/bin/env python3
"""
Main training script for Hierarchical Attention MIL model
"""
import os
import time
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
import glob
import numpy as np

EMB_DIR = "/projects/e32998/patches_varsize_pooled4096"  # <-- same as precompute script output

# Import our modules
from config import DATA_PATHS, TRAINING_CONFIG, MODEL_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice,
    build_slice_to_class_map, split_by_case_stratified, build_case_dict,
    report_no_leak, summarize_case_dict
)
from models import create_model
from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
from trainer import MILTrainer, count_patches_by_class
from utils import (
    set_seed, get_device, print_data_summary, create_run_directory,
    save_data_splits, load_data_splits, print_model_summary, check_data_integrity
)
from attention_analysis import analyze_attention_weights


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Hierarchical Attention MIL model')
    
    # Data arguments
    parser.add_argument('--labels_csv', type=str, default=DATA_PATHS['labels_csv'],
                       help='Path to labels CSV file')
    parser.add_argument('--patches_dir', type=str, default=DATA_PATHS['patches_dir'],
                       help='Path to patches directory')
    # checkpoint_dir is now automatically set to {run_dir}/checkpoints
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help='Batch size (typically 1 for MIL)')
    parser.add_argument('--num_workers', type=int, default=TRAINING_CONFIG['num_workers'],
                       help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=MODEL_CONFIG['embed_dim'],
                       help='Embedding dimension')
    parser.add_argument('--per_slice_cap', type=int, default=MODEL_CONFIG['per_slice_cap'],
                       help='Maximum patches per slice')
    parser.add_argument('--max_slices_per_stain', type=int, default=MODEL_CONFIG['max_slices_per_stain'],
                       help='Maximum slices per stain (None for unlimited)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['random_state'],
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not train')
    parser.add_argument('--analyze_attention', action='store_true',
                       help='Perform attention analysis and visualization')
    parser.add_argument('--attention_top_n', type=int, default=5,
                       help='Number of top/bottom patches to visualize')
    parser.add_argument('--load_splits', type=str, default=None,
                       help='Path to data_splits.npz file to load existing splits')
    parser.add_argument('--multi_run', action='store_true',
                       help='Run all split files across multiple seeds and save aggregate CSVs')
    parser.add_argument('--splits_dir', type=str, default=None,
                       help='Directory containing split .npz files')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Seeds to use in multi-run mode')
    parser.add_argument('--aggregate_output_dir', type=str, default=None,
                       help='Directory for aggregate CSV outputs')
    parser.add_argument('--split_name', type=str, default=None,
                       help='Optional split name for output labeling')
    
    return parser.parse_args()


def prepare_data(args):
    """Prepare and split the data"""
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    # Load labels
    labels = load_labels(args.labels_csv)
    print(f"Loaded {len(labels)} labels")
    
    # Get patch files
    all_files = get_all_patch_files(args.patches_dir)
    print(f"Found {len(all_files)} patch files")
    
    # Group patches by slice
    patches = group_patches_by_slice(all_files, args.patches_dir)
    print(f"Grouped into {len(patches)} slices")
    
    # Build slice to class mapping
    slice_to_class = build_slice_to_class_map(patches, labels)
    print(f"Mapped {len(slice_to_class)} slices to classes")
    
    # Group slices by class for stratified splitting
    slices_by_class = defaultdict(list)
    for key, label in slice_to_class.items():
        slices_by_class[label].append(key)
    
    print(f"Class distribution: {dict((k, len(v)) for k, v in slices_by_class.items())}")
    
    print("\n" + "-" * 40)
    print("SPLITTING DATA")
    print("-" * 40)
    
    if args.load_splits:
        # Load existing splits
        print(f"Loading existing splits from: {args.load_splits}")
        splits_data = load_data_splits(args.load_splits)
        train_cases_set = set(splits_data['train_cases'])
        val_cases_set = set(splits_data['val_cases'])
        test_cases_set = set(splits_data['test_cases'])
        
        # Map loaded case IDs back to slices
        train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in train_cases_set]
        val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in val_cases_set]
        test_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in test_cases_set]
        
        print(f"Loaded splits - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
    else:
        # Split data by case (stratified)
        train_slices, val_slices, test_slices = split_by_case_stratified(
            slices_by_class, random_state=args.seed
        )
        
        print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
    
    # Build case dictionaries
    train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
    val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
    test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)
    
    # Check for data leakage
    report_no_leak(train_case_dict, val_case_dict, test_case_dict)
    
    # Create summary DataFrames
    train_df = summarize_case_dict(train_case_dict, train_label_map, "train")
    val_df = summarize_case_dict(val_case_dict, val_label_map, "val")
    test_df = summarize_case_dict(test_case_dict, test_label_map, "test")
    
    # Print data summary
    print_data_summary(train_df, val_df, test_df)
    
    # Count patches by class
    count_patches_by_class(train_case_dict, train_label_map, "Train")
    count_patches_by_class(val_case_dict, val_label_map, "Validation")
    count_patches_by_class(test_case_dict, test_label_map, "Test")
    
    # Check data integrity
    check_data_integrity(train_case_dict, train_label_map, "Train")
    check_data_integrity(val_case_dict, val_label_map, "Validation")
    check_data_integrity(test_case_dict, test_label_map, "Test")
    
    return (train_case_dict, train_label_map), (val_case_dict, val_label_map), (test_case_dict, test_label_map)


def create_data_loaders(train_data, val_data, test_data, args):
    """Create data loaders (precomputed pooled features)"""
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS (POOLED FEATURES)")
    print("=" * 60)

    train_case_dict, train_label_map = train_data
    val_case_dict, val_label_map = val_data
    test_case_dict, test_label_map = test_data

    train_ds = StainBagCasePooledFeatureDataset(
        train_case_dict, train_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=True,
    )

    val_ds = StainBagCasePooledFeatureDataset(
        val_case_dict, val_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=False,
    )

    test_ds = StainBagCasePooledFeatureDataset(
        test_case_dict, test_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    print(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader

# new helper functions
def discover_split_files(splits_dir):
    split_files = sorted(glob.glob(os.path.join(splits_dir, "*.npz")))
    if not split_files:
        raise FileNotFoundError(f"No .npz split files found in {splits_dir}")
    return split_files


def get_split_id(split_path, idx=None):
    base = os.path.basename(split_path)
    stem = os.path.splitext(base)[0]
    if idx is None:
        return stem
    return f"split_{idx+1}_{stem}"


def results_to_prediction_df(test_results, split_id, seed):
    """
    One row per (split, case, seed).
    This is boxplot-ready for predicted probabilities.
    """
    rows = []
    probs_list = test_results["prediction_probs"]

    for i, case_id in enumerate(test_results["case_ids"]):
        probs = probs_list[i]

        row = {
            "split_id": split_id,
            "case_id": case_id,
            "seed": seed,
            "prob_class1": float(probs[1]) if len(probs) > 1 else float(probs[0]),
            "correct": int(test_results["true_labels"][i] == test_results["predictions"][i]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_and_save(all_run_rows, all_prediction_dfs, output_dir):
    """
    Save only the assignment-relevant CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    run_summary_df = pd.DataFrame(all_run_rows)
    predictions_long_df = (
        pd.concat(all_prediction_dfs, ignore_index=True)
        if all_prediction_dfs else pd.DataFrame(columns=["split_id", "case_id", "seed", "prob_class1", "correct"])
    )

    # 1) Best validation loss boxplot-ready
    best_val_loss_df = run_summary_df[["split_id", "seed", "best_val_loss"]].copy()
    best_val_loss_df.to_csv(
        os.path.join(output_dir, "best_val_loss_boxplot_ready.csv"),
        index=False
    )

    # 2) Predicted probabilities boxplot-ready
    prob_boxplot_df = predictions_long_df[["split_id", "case_id", "seed", "prob_class1"]].copy()
    prob_boxplot_df.to_csv(
        os.path.join(output_dir, "predicted_probabilities_boxplot_ready.csv"),
        index=False
    )

    # 3) Classification consistency: per (split, case), NOT across splits
    consistency_df = (
        predictions_long_df
        .groupby(["split_id", "case_id"], as_index=False)
        .agg(pct_correct=("correct", lambda x: 100.0 * float(np.mean(x))))
    )
    consistency_df.to_csv(
        os.path.join(output_dir, "classification_consistency.csv"),
        index=False
    )

    # 4) Accuracy variability: per split only
    accuracy_variability_df = (
        run_summary_df
        .groupby("split_id", as_index=False)
        .agg(
            min_test_accuracy=("test_accuracy", "min"),
            max_test_accuracy=("test_accuracy", "max"),
        )
    )
    accuracy_variability_df.to_csv(
        os.path.join(output_dir, "accuracy_variability.csv"),
        index=False
    )

    print(f"\nAggregate CSVs saved to: {output_dir}")
    print("  - best_val_loss_boxplot_ready.csv")
    print("  - predicted_probabilities_boxplot_ready.csv")
    print("  - classification_consistency.csv")
    print("  - accuracy_variability.csv")

def run_one(args, split_path=None, split_id="single_run"):
    """
    Run one training/eval job for one fixed split and one seed.
    """
    start_time = time.time()

    set_seed(args.seed)
    device = get_device()

    run_dir = create_run_directory()
    args.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if split_path is not None:
        args.load_splits = split_path

    print("=" * 80)
    print("HIERARCHICAL ATTENTION MIL TRAINING")
    print("=" * 80)
    print(f"Split: {split_id}")
    print(f"Seed: {args.seed}")
    print(f"Arguments: {vars(args)}")

    train_data, val_data, test_data = prepare_data(args)

    train_cases = list(train_data[0].keys())
    val_cases = list(val_data[0].keys())
    test_cases = list(test_data[0].keys())

    if not args.load_splits:
        save_data_splits(train_cases, val_cases, test_cases, run_dir)

    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, args)

    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    model = create_model(
        num_classes=MODEL_CONFIG['num_classes'],
        embed_dim=args.embed_dim,
    )
    print_model_summary(model)

    trainer = MILTrainer(model, device, checkpoint_dir=args.checkpoint_dir)

    if args.lr != TRAINING_CONFIG['learning_rate']:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Updated learning rate to {args.lr}")

    start_epoch = 0
    if args.resume:
        print("\n" + "-" * 40)
        print("LOADING CHECKPOINT")
        print("-" * 40)
        start_epoch = trainer.load_checkpoint(args.resume)

    if not args.eval_only:
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        train_info = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            start_epoch=start_epoch
        )
    else:
        train_info = {
            "best_val_loss": trainer.best_val_loss,
            "best_checkpoint_path": args.resume,
        }

    # Evaluate best checkpoint, not just last epoch
    best_ckpt = train_info.get("best_checkpoint_path", None)
    if best_ckpt is not None and os.path.exists(best_ckpt):
        print(f"\nReloading best checkpoint before test evaluation: {best_ckpt}")
        trainer.load_checkpoint(best_ckpt)

    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    test_results = trainer.evaluate(
        test_loader, 
        save_predictions=True, 
        output_dir=run_dir,
        checkpoint_name=best_ckpt if best_ckpt else None
    )

    run_row = {
        "split_id": split_id,
        "seed": args.seed,
        "best_val_loss": float(train_info["best_val_loss"]),
        "test_accuracy": float(test_results["test_accuracy"]),
    }

    prediction_df = results_to_prediction_df(test_results, split_id, args.seed)

    total_time = time.time() - start_time
    print(f"\nFinished split={split_id}, seed={args.seed} in {total_time:.2f}s")

    return run_row, prediction_df

def main():
    """Main training function"""
    args = parse_args()

    if args.multi_run:
        if args.splits_dir is None:
            raise ValueError("--splits_dir is required with --multi_run")
        if args.seeds is None or len(args.seeds) == 0:
            raise ValueError("--seeds is required with --multi_run")
        if args.aggregate_output_dir is None:
            raise ValueError("--aggregate_output_dir is required with --multi_run")
    
        split_files = discover_split_files(args.splits_dir)

        all_run_rows = []
        all_prediction_dfs = []

        for i, split_path in enumerate(split_files):
            split_id = get_split_id(split_path, i)

            for seed in args.seeds:
                print(f"\nRunning {split_id} | seed={seed}")

                run_args = argparse.Namespace(**vars(args))
                run_args.seed = seed

                run_row, pred_df = run_one(run_args, split_path, split_id)

                all_run_rows.append(run_row)
                all_prediction_dfs.append(pred_df)

        aggregate_and_save(
            all_run_rows,
            all_prediction_dfs,
            args.aggregate_output_dir
        )

        return

    start_time = time.time()
    # Set up
    set_seed(args.seed)
    device = get_device()
    
    # Create run directory
    run_dir = create_run_directory()
    
    # Update checkpoint directory to run directory
    args.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 80)
    print("HIERARCHICAL ATTENTION MIL TRAINING")
    print("=" * 80)
    print(f"Arguments: {vars(args)}")
    print(f"Training config (may be overridden by arguments): {TRAINING_CONFIG}")
    
    # Prepare data
    train_data, val_data, test_data = prepare_data(args)
    
    # Save data splits for reproducibility (unless loaded from existing)
    train_cases = list(train_data[0].keys())
    val_cases = list(val_data[0].keys())
    test_cases = list(test_data[0].keys())
    if not args.load_splits:
        save_data_splits(train_cases, val_cases, test_cases, run_dir)
    else:
        print(f"Using existing splits from: {args.load_splits}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, args)
    
    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    model = create_model(
        num_classes=MODEL_CONFIG['num_classes'],
        embed_dim=args.embed_dim,
    )
    print_model_summary(model)
    
    # Create trainer
    trainer = MILTrainer(model, device, checkpoint_dir=args.checkpoint_dir)
    
    # Update trainer learning rate if different from config
    if args.lr != TRAINING_CONFIG['learning_rate']:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Updated learning rate to {args.lr}")
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        print("\n" + "-" * 40)
        print("LOADING CHECKPOINT")
        print("-" * 40)
        start_epoch = trainer.load_checkpoint(args.resume)
    
    if not args.eval_only:
        # Train the model
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        train_info = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            start_epoch=start_epoch
        )
    else:
        train_info = {
            "best_val_loss": trainer.best_val_loss,
            "best_checkpoint_path": args.resume,
        }

    best_ckpt = train_info.get("best_checkpoint_path", None)
    if best_ckpt is not None and os.path.exists(best_ckpt):
        print(f"\nReloading best checkpoint before test evaluation: {best_ckpt}")
        trainer.load_checkpoint(best_ckpt)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    test_results = trainer.evaluate(
        test_loader, 
        save_predictions=True, 
        output_dir=run_dir,
        checkpoint_name=best_ckpt if best_ckpt else None
    )
    
    # Attention analysis if requested
    if args.analyze_attention:
        analyze_attention_weights(
            trainer.model, 
            test_loader, 
            run_dir, 
            top_n=args.attention_top_n
        )
    
    # Save final results
    results_path = os.path.join(run_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"Test Loss: {test_results['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {test_results['test_accuracy']:.4f}\n")
        f.write(f"Number of samples: {test_results['num_samples']}\n")
        if args.resume:
            f.write(f"Checkpoint used: {args.resume}\n")
        f.write(f"\nOutput files:\n")
        f.write(f"- predictions.csv: Per-case predictions and probabilities\n")
        f.write(f"- confusion_matrix.png: Visual confusion matrix\n")
        if args.analyze_attention:
            f.write(f"- attention_analysis/: Attention visualizations and summary\n")
    
    print(f"\nResults saved to: {run_dir}")
    total_time = time.time() - start_time
    hours = total_time / 3600
    print(f"\nTotal execution time: {total_time:.2f}s ({hours:.2f} hours)")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()