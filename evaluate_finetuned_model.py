"""
Evaluate the fine-tuned GPT-4o model for data center detection.
"""

import os
import json
import base64
import random
import argparse
from pathlib import Path
from openai import OpenAI
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set API key
api_key = "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"
client = OpenAI(api_key=api_key)

# Configuration
FINETUNED_MODEL = "ft:gpt-4o-2024-08-06:redwood-research::CLf6NRke"
DATA_CENTER_DIR = "classifier_data/eval/data_center"
NON_DATA_CENTER_DIR = "classifier_data/eval/non_data_center"
TEST_SIZE = None  # Use all eval images (set to None for all)
# MAX_IMAGE_SIZE = (512, 512)
DEBUG_LOGPROBS = False  # Set to True to debug probability calculation
MAX_WORKERS = 20  # Number of concurrent threads


def encode_image_to_base64(image_path):
    """Encode image file to base64 string, with resizing."""
    img = Image.open(image_path)
    # img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def classify_image(image_path, model, return_logprobs=False):
    """Classify a single image using the specified model."""
    base64_image = encode_image_to_base64(image_path)
    image_url = f"data:image/png;base64,{base64_image}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a system that helps me with climate planning by identifying whether buildings are data centers."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Is this a data center?" 
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        max_tokens=10,
        logprobs=return_logprobs,
        top_logprobs=20 if return_logprobs else None
    )

    answer = response.choices[0].message.content.strip().lower()
    print(answer)
    predicted = "yes" in answer

    if return_logprobs:
        # Extract logprobs for first token (should be "yes" or "no")
        logprobs_data = response.choices[0].logprobs.content[0].top_logprobs
        return predicted, logprobs_data

    return predicted


def compute_metrics_from_results(results_list):
    """Compute accuracy, precision, recall, F1 from a list of evaluation results."""
    correct = 0
    total = 0
    confusion = {
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }

    for result in results_list:
        if result['error']:
            continue

        total += 1
        is_data_center = result['is_data_center']
        predicted = result['predicted']

        if predicted == is_data_center:
            correct += 1
            if is_data_center:
                confusion['true_positives'] += 1
            else:
                confusion['true_negatives'] += 1
        else:
            if is_data_center:
                confusion['false_negatives'] += 1
            else:
                confusion['false_positives'] += 1

    accuracy = correct / total if total > 0 else 0
    precision = confusion['true_positives'] / (confusion['true_positives'] + confusion['false_positives']) if (confusion['true_positives'] + confusion['false_positives']) > 0 else 0
    recall = confusion['true_positives'] / (confusion['true_positives'] + confusion['false_negatives']) if (confusion['true_positives'] + confusion['false_negatives']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion,
        'total': total,
        'correct': correct
    }


def evaluate_single_image_with_logprobs(image_path, is_data_center, model_name):
    """Evaluate a single image with logprobs."""
    import numpy as np

    try:
        predicted, logprobs = classify_image(image_path, model_name, return_logprobs=True)

        # Extract probability for "yes" and "no" tokens
        yes_logprob = None
        no_logprob = None

        for token_data in logprobs:
            token = token_data.token.strip().lower()
            if token in ["yes", "y"]:
                if yes_logprob is None or token_data.logprob > yes_logprob:
                    yes_logprob = token_data.logprob
            elif token in ["no", "n"]:
                if no_logprob is None or token_data.logprob > no_logprob:
                    no_logprob = token_data.logprob

        # Calculate probability of "yes"
        if yes_logprob is not None and no_logprob is not None:
            prob_yes = np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
        elif yes_logprob is not None and no_logprob is None:
            prob_yes = np.exp(yes_logprob)
        elif no_logprob is not None and yes_logprob is None:
            prob_yes = np.exp(no_logprob - 10)
        else:
            prob_yes = 0.5

        return {
            'image_path': image_path,
            'is_data_center': is_data_center,
            'prob_yes': prob_yes,
            'predicted': predicted,
            'yes_logprob': yes_logprob,
            'no_logprob': no_logprob,
            'logprobs': logprobs,
            'error': None
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'is_data_center': is_data_center,
            'prob_yes': 0.5,
            'predicted': None,
            'yes_logprob': None,
            'no_logprob': None,
            'logprobs': None,
            'error': str(e)
        }


def evaluate_model_with_logprobs(model_name, test_images):
    """Evaluate model and collect logprobs in one pass."""
    import numpy as np

    print(f"\nEvaluating {model_name}...")
    print("-" * 60)

    results_list = []

    # Run evaluations concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_image = {
            executor.submit(evaluate_single_image_with_logprobs, img_path, is_dc, model_name): (img_path, is_dc)
            for img_path, is_dc in test_images
        }

        completed = 0
        for future in as_completed(future_to_image):
            completed += 1
            result = future.result()
            results_list.append(result)

            image_path = result['image_path']
            is_data_center = result['is_data_center']
            predicted = result['predicted']
            prob_yes = result['prob_yes']

            print(f"[{completed}/{len(test_images)}] {image_path.name}... ", end="")

            if result['error']:
                print(f"Error: {result['error']}")
                continue

            correct = predicted == is_data_center
            if correct:
                if is_data_center:
                    print("✓ Correct (True Positive)")
                else:
                    print("✓ Correct (True Negative)")
            else:
                if is_data_center:
                    print("✗ Wrong (False Negative - missed data center)")
                else:
                    print("✗ Wrong (False Positive - false alarm)")

            if DEBUG_LOGPROBS:
                print(f"  score={prob_yes:.3f}, yes_lp={result['yes_logprob']}, no_lp={result['no_logprob']}")
                if result['logprobs']:
                    print(f"  All tokens: {[(lp.token, lp.logprob, lp.token.strip().lower()) for lp in result['logprobs'][:10]]}")

    # Compute metrics
    metrics = compute_metrics_from_results(results_list)

    print("\n" + "=" * 60)
    print(f"Results for {model_name}")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['confusion_matrix']['true_positives']}")
    print(f"  True Negatives:  {metrics['confusion_matrix']['true_negatives']}")
    print(f"  False Positives: {metrics['confusion_matrix']['false_positives']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['false_negatives']}")

    return metrics, results_list


def generate_roc_curve(model_name, results_list):
    """Generate ROC curve from previously collected results with logprobs."""
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    print(f"\nGenerating ROC curve for {model_name}...")
    print("-" * 60)

    y_true = []
    y_scores = []

    for result in results_list:
        if result['error']:
            continue

        y_true.append(1 if result['is_data_center'] else 0)
        y_scores.append(result['prob_yes'])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save plot
    filename = f"roc_curve_{model_name.replace(':', '_').replace('/', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to {filename}")
    print(f"AUC: {roc_auc:.3f}")

    return roc_auc


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned GPT-4o model for data center detection')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to test per class (default: all images)')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline model evaluation')
    parser.add_argument('--skip-roc', action='store_true',
                        help='Skip ROC curve generation')
    args = parser.parse_args()

    # Override TEST_SIZE with command line argument if provided
    test_size = args.num_images if args.num_images is not None else TEST_SIZE

    print("GPT-4o Fine-Tuned Model Evaluation")
    print("=" * 60)

    # Collect test images
    test_images = []

    # Sample data center images
    dc_path = Path(DATA_CENTER_DIR)
    if dc_path.exists():
        all_dc = list(dc_path.glob("*.png"))
        if test_size is None:
            sampled_dc = all_dc
        else:
            sampled_dc = random.sample(all_dc, min(test_size, len(all_dc)))
        test_images.extend([(img, True) for img in sampled_dc])
        print(f"Selected {len(sampled_dc)} data center test images")

    # Sample non-data center images
    non_dc_path = Path(NON_DATA_CENTER_DIR)
    if non_dc_path.exists():
        all_non_dc = list(non_dc_path.glob("*.png"))
        if test_size is None:
            sampled_non_dc = all_non_dc
        else:
            sampled_non_dc = random.sample(all_non_dc, min(test_size, len(all_non_dc)))
        test_images.extend([(img, False) for img in sampled_non_dc])
        print(f"Selected {len(sampled_non_dc)} non-data center test images")

    # Shuffle test images
    random.shuffle(test_images)

    print(f"\nTotal test images: {len(test_images)}")

    # Evaluate finetuned model (with logprobs)
    finetuned_metrics, finetuned_results_list = evaluate_model_with_logprobs(FINETUNED_MODEL, test_images)

    # Save results to JSON file
    results_data = {
        'model': FINETUNED_MODEL,
        'metrics': finetuned_metrics,
        'results': [
            {
                'image_path': str(r['image_path']),
                'is_data_center': r['is_data_center'],
                'predicted': r['predicted'],
                'prob_yes': r['prob_yes'],
                'yes_logprob': r['yes_logprob'],
                'no_logprob': r['no_logprob'],
                'error': r['error']
            }
            for r in finetuned_results_list
        ]
    }

    output_file = f"evaluation_results_{FINETUNED_MODEL.replace(':', '_').replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Generate ROC curve from the same results
    if not args.skip_roc:
        print("\n" + "=" * 60)
        print("GENERATING ROC CURVE")
        print("=" * 60)
        finetuned_auc = generate_roc_curve(FINETUNED_MODEL, finetuned_results_list)

        print("\n" + "=" * 60)
        print("ROC AUC")
        print("=" * 60)
        print(f"Fine-tuned AUC: {finetuned_auc:.3f}")


if __name__ == "__main__":
    main()
