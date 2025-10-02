#!/usr/bin/env python3
"""
Analyze saved evaluation results and generate ROC curves with additional metrics.
"""

import json
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


def load_results(filepath):
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_manual_labels(manual_labels_path, false_negatives_path=None):
    """Load manual labels from JSON file and map to image paths."""
    if not os.path.exists(manual_labels_path):
        return {}

    with open(manual_labels_path, 'r') as f:
        manual_labels = json.load(f)

    # Create a dictionary mapping location (coordinates) to manually determined label
    labels_by_coords = {}
    for item in manual_labels:
        location = item['location']
        labels_by_coords[location] = item['manually_determined_label']

    # Load false negatives to get image_path to coordinates mapping
    image_to_coords = {}
    if false_negatives_path and os.path.exists(false_negatives_path):
        with open(false_negatives_path, 'r') as f:
            false_negs = json.load(f)
        for item in false_negs.get('misclassifications', []):
            if 'coordinates' in item and 'image_path' in item:
                image_to_coords[item['image_path']] = item['coordinates']

    # Create mapping from image_path to manually determined label
    labels_by_image = {}
    for image_path, coords in image_to_coords.items():
        if coords in labels_by_coords:
            labels_by_image[image_path] = labels_by_coords[coords]

    return labels_by_image


def compute_metrics(results_data, manual_labels_path=None, false_negatives_path=None):
    """Compute metrics from saved results, correcting ground truth with manual labels."""
    results = results_data['results']

    # Load manual labels if provided
    manual_labels = {}
    if manual_labels_path:
        manual_labels = load_manual_labels(manual_labels_path, false_negatives_path)

    # Extract ground truth and predictions
    y_true = []
    y_scores = []
    y_pred = []
    excluded_count = 0

    for r in results:
        if r['error']:
            continue

        image_path = r['image_path']

        # Check if this image has a manual label correction
        if image_path in manual_labels:
            manual_label = manual_labels[image_path]
            # Exclude items that are not actually data centers or have image issues
            if manual_label in ['not_data_center', 'image_cut_off']:
                excluded_count += 1
                continue
            # Use the manually corrected label
            is_data_center = (manual_label == 'data_center')
        else:
            is_data_center = r['is_data_center']

        y_true.append(1 if is_data_center else 0)
        y_scores.append(r['prob_yes'])
        y_pred.append(1 if r['predicted'] else 0)

    if excluded_count > 0:
        print(f"\nExcluded {excluded_count} samples based on manual labels (not_data_center or image_cut_off)")

    if manual_labels:
        print(f"Applied manual label corrections for {len(manual_labels)} locations")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)

    # Compute confusion matrix
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Compute basic metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Find TPR at FPR = 1%
    target_fpr = 0.01
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) > 0:
        tpr_at_1pct_fpr = tpr[idx[-1]]
        threshold_at_1pct_fpr = thresholds[idx[-1]]
    else:
        tpr_at_1pct_fpr = 0.0
        threshold_at_1pct_fpr = None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'tpr_at_1pct_fpr': tpr_at_1pct_fpr,
        'threshold_at_1pct_fpr': threshold_at_1pct_fpr,
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        },
        'roc_data': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }


def plot_roc_curve(results_data, metrics, output_file='roc_curve.png'):
    """Plot ROC curve with metrics."""
    model_name = results_data['model']
    fpr = np.array(metrics['roc_data']['fpr'])
    tpr = np.array(metrics['roc_data']['tpr'])
    auc_score = metrics['auc']
    tpr_at_1pct = metrics['tpr_at_1pct_fpr']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Linear scale plot
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    ax1.plot([0.01], [tpr_at_1pct], 'ro', markersize=10,
             label=f'TPR @ FPR=1%: {tpr_at_1pct:.1%}')
    ax1.axvline(x=0.01, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curve (Linear Scale)', fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # Log scale plot
    # Filter out zeros for log scale
    fpr_nonzero = fpr[fpr > 0]
    tpr_nonzero = tpr[fpr > 0]

    ax2.plot(fpr_nonzero, tpr_nonzero, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.3f})')
    ax2.plot([0.01], [tpr_at_1pct], 'ro', markersize=10,
             label=f'TPR @ FPR=1%: {tpr_at_1pct:.1%}')
    ax2.axvline(x=0.01, color='red', linestyle=':', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlim([max(fpr_nonzero.min(), 1e-4), 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (log scale)', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title(f'ROC Curve (Log Scale)', fontsize=14)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    plt.suptitle(f'ROC Analysis - {model_name}', fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved to {output_file}")
    plt.close()


def print_metrics(metrics):
    """Print all metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)

    print(f"\nAccuracy:  {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    print(f"AUC:       {metrics['auc']:.3f}")

    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Positives:  {cm['tp']}")
    print(f"  True Negatives:  {cm['tn']}")
    print(f"  False Positives: {cm['fp']}")
    print(f"  False Negatives: {cm['fn']}")

    print(f"\nROC Curve Metrics:")
    print(f"  TPR @ FPR = 1%:  {metrics['tpr_at_1pct_fpr']:.1%}")
    if metrics['threshold_at_1pct_fpr'] is not None:
        print(f"  Threshold:       {metrics['threshold_at_1pct_fpr']:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze saved evaluation results')
    parser.add_argument('input_file', help='Path to evaluation results JSON file')
    parser.add_argument('--output', '-o', default='roc_curve_analysis.png',
                        help='Output filename for ROC curve plot')
    parser.add_argument('--manual-labels', '-m', default=None,
                        help='Path to manual labels JSON file for correcting ground truth')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.input_file}...")
    results_data = load_results(args.input_file)

    print(f"Model: {results_data['model']}")
    print(f"Total results: {len(results_data['results'])}")

    # If manual labels path not provided, try to find it in a standard location
    manual_labels_path = args.manual_labels
    false_negatives_path = None
    if manual_labels_path is None:
        # Try to find manual_labels.json in misclassified/ subdirectory
        input_dir = os.path.dirname(args.input_file)
        if not input_dir:
            input_dir = '.'
        potential_path = os.path.join(input_dir, 'misclassified', 'manual_labels.json')
        if os.path.exists(potential_path):
            manual_labels_path = potential_path
            print(f"\nAuto-detected manual labels at: {manual_labels_path}")

        # Also look for false_negatives.json
        fn_path = os.path.join(input_dir, 'misclassified', 'false_negatives.json')
        if os.path.exists(fn_path):
            false_negatives_path = fn_path
            print(f"Auto-detected false negatives at: {false_negatives_path}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results_data, manual_labels_path, false_negatives_path)

    # Print metrics
    print_metrics(metrics)

    # Plot ROC curve
    print(f"\nGenerating ROC curve plot...")
    plot_roc_curve(results_data, metrics, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
