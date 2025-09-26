import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data for relevant features
data = {
    "Model": ["YOLOv5", "SSD", "Faster R-CNN", "YOLOv8"],
    "Accuracy": [0.89, 0.85, 0.88, 0.92],
    "Precision": [0.87, 0.84, 0.86, 0.91],
    "Recall": [0.88, 0.83, 0.87, 0.93],
    "Model Size (MB)": [150, 120, 200, 90],
    "Utilization (%)": [75, 70, 80, 85]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Calculate correlation matrix (excluding the 'Model' column as it's categorical)
correlation_matrix = df.drop("Model", axis=1).corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Adjust figure size
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    annot_kws={"size": 10},  # Annotation size
    linewidths=0.5  # Add grid lines for clarity
)

# Customize axis labels
plt.title("Correlation Heatmap of Object Detection Features", fontsize=14, weight='bold', pad=20)
plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)  # Adjust y-axis labels font size

# Save the plot if needed
plt.tight_layout()  # Ensure everything fits within the figure
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")  # Save the image

# Show the heatmap
plt.show()
