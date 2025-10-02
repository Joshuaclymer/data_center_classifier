import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the heuristic analysis results
with open('../heuristic_analysis_results.json', 'r') as f:
    data = json.load(f)

# Extract property 6 scores and true labels
property_6_scores = []
true_labels = []

for result in data['results']:
    if 'heuristic_analysis' in result and 'property_6' in result['heuristic_analysis']:
        property_6_scores.append(result['heuristic_analysis']['property_6'])
        # Convert type to binary: 1 if data_center, 0 otherwise
        true_labels.append(1 if result['type'] == 'data_center' else 0)

# Convert to numpy arrays
property_6_scores = np.array(property_6_scores)
true_labels = np.array(true_labels)

print(f"Total samples: {len(property_6_scores)}")
print(f"Data centers: {sum(true_labels)}")
print(f"Non-data centers: {len(true_labels) - sum(true_labels)}")
print(f"Property 6 score range: {min(property_6_scores)} to {max(property_6_scores)}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, property_6_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Property 6: "Is a data center"')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_property6.png', dpi=300, bbox_inches='tight')
# plt.show()

print(f"\nROC AUC Score: {roc_auc:.3f}")