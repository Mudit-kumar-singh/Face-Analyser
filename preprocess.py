import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

age_gender_data = r"UTKface DATA\UTKFace"
SAVE_DIR = "Processed_Data"

os.makedirs(SAVE_DIR, exist_ok=True)

images = []
gender_labels = []
age_labels = []
skipped = 0

for img in tqdm(os.listdir(age_gender_data), desc="Preprocessing Data"):
    img_path = os.path.join(age_gender_data, img)
    img_array = cv.imread(img_path)
    if img_array is None:
        skipped += 1
        continue    
    parts = img.split('_')
    try:    
        age = int(parts[0])
        gender = int(parts[1])
        img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
        img_resized = cv.resize(img_rgb, (128, 128))

        images.append(img_resized)
        gender_labels.append(gender)
        age_labels.append(age)
    except:
        skipped += 1
        continue
X = np.array(images,dtype=np.float32)/255.0
age_labels = np.array(age_labels, dtype=int)
gender_labels = np.array(gender_labels, dtype=int)

np.save(os.path.join(SAVE_DIR, "X.npy"),X)
np.save(os.path.join(SAVE_DIR, "age_labels.npy"),age_labels)
np.save(os.path.join(SAVE_DIR, "gender_labels.npy"),gender_labels)
print(f" Saved! Images: {len(images)}, Skipped: {skipped}")






