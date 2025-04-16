import matplotlib.pyplot as plt

# Data from Section 3.1
models = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']
accuracy = [85.2, 82.7, 78.9, 87.6]
precision = [84.5, 81.3, 76.5, 86.8]
recall = [83.8, 80.9, 77.2, 85.9]
f1_score = [84.1, 81.1, 76.8, 86.3]
roc_auc = [89, 87, 81, 91]  # ROC-AUC scaled to percentages for consistency

# Create a line plot
plt.figure(figsize=(10, 6))

# Plot each metric
plt.plot(models, accuracy, marker='o', label='Accuracy', linestyle='-')
plt.plot(models, precision, marker='s', label='Precision', linestyle='--')
plt.plot(models, recall, marker='^', label='Recall', linestyle='-.')
plt.plot(models, f1_score, marker='D', label='F1-Score', linestyle=':')
plt.plot(models, roc_auc, marker='x', label='ROC-AUC', linestyle='-')

# Add labels, title, and legend
plt.xlabel('Models', fontsize=12)
plt.ylabel('Performance Metrics (%)', fontsize=12)
plt.title('Performance Metrics of Classification Models', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
