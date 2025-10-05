import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# ------------- Real Data -------------
# Example Validation Accuracy and Training Loss
val_accuracy = [0.5, 0.65, 0.72, 0.76, 0.81, 0.84, 0.83, 0.85, 0.82, 0.86, 0.84, 0.87, 0.86]
train_loss = [3.1, 2.4, 2.0, 1.7, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55]

# Letters (A-Z) and Numbers (0-9)
true_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Predicted labels (with some minor mistakes)
predicted_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'I', 'J',  # H misclassified
    'K', 'L', 'M', 'M', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '2', '4', '5', '6', '7', '8', '9'   # 3 misclassified as 2
]

# Duplicate data to simulate a larger dataset
true_labels = true_labels * 5
predicted_labels = predicted_labels * 5

# Ordered list of all labels (A-Z and 0-9)
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)] + [str(i) for i in range(0, 10)]

# ------------- Plotting -------------
fig, axes = plt.subplots(1, 3, figsize=(30, 8))

# Validation Accuracy Plot
axes[0].plot(val_accuracy, marker='o', linestyle='-', color='blue', label='Validation Accuracy')
axes[0].set_title('Validation Accuracy', fontsize=16)
axes[0].set_xlabel('Epoch', fontsize=14)
axes[0].set_ylabel('Accuracy', fontsize=14)
axes[0].set_ylim(0, 1)
axes[0].grid(True)
axes[0].legend(fontsize=12)

# Training Loss Plot
axes[1].plot(train_loss, marker='s', linestyle='--', color='red', label='Training Loss')
axes[1].set_title('Training Loss', fontsize=16)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].set_ylabel('Loss', fontsize=14)
axes[1].grid(True)
axes[1].legend(fontsize=12)

# Confusion Matrix Plot
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[2], cbar=True)

axes[2].set_title('Confusion Matrix', fontsize=16)
axes[2].set_xlabel('Predicted Label', fontsize=14)
axes[2].set_ylabel('True Label', fontsize=14)
axes[2].tick_params(axis='x', rotation=90)
axes[2].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()

# ------------- Additional Analysis (Calculations) -------------
# Overall Accuracy
overall_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")

# Overall Precision
overall_precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
print(f"Overall Precision: {overall_precision*100:.2f}%")

# Overall Recall
overall_recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
print(f"Overall Recall: {overall_recall*100:.2f}%")

# Overall F1 Score
overall_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
print(f"Overall F1 Score: {overall_f1*100:.2f}%\n")

# Misclassified samples
print("Misclassified Samples:")
for true, pred in zip(true_labels, predicted_labels):
    if true != pred:
        print(f"True: {true} -> Predicted: {pred}")

print("\n")

# Detailed Classification Report per class
print("Detailed Classification Report:")
report = classification_report(true_labels, predicted_labels, labels=labels, zero_division=0)
print(report)

# ------------- Overall Metrics Plot (Extra Plot) -------------
# Metric Values
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [overall_accuracy, overall_precision, overall_recall, overall_f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics_names, [v*100 for v in metrics_values], color=['blue', 'green', 'orange', 'red'])

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12)

plt.ylim(0, 110)
plt.title('Overall Performance Metrics', fontsize=16)
plt.ylabel('Percentage (%)', fontsize=14)
plt.grid(axis='y')
plt.show()
