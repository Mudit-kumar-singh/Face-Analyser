import numpy as np

X = np.load("Processed_Data/X.npy")

X = (X * 255).astype(np.uint8)

np.save("Processed_Data/X_uint8.npy", X)

print("Shape:", X.shape)
print("Dtype:", X.dtype)
print("Size:", X.nbytes / (1024**3), "GB")