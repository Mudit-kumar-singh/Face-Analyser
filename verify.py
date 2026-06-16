import numpy as np

X = np.load("Processed_Data/X.npy")
age_labels = np.load("Processed_Data/age_labels.npy")
gender_labels = np.load("Processed_Data/gender_labels.npy")

print("Shape:", X.shape)
print("Dtype:", X.dtype)
print("Min value:", X.min())
print("Max value:", X.max())
print("Age range:", age_labels.min(), "-", age_labels.max())
print("Gender unique values:", np.unique(gender_labels))