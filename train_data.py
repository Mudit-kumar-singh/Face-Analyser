import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from google.colab import drive

drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/face analyser data'

# Load labels only
age_labels    = np.load(f'{BASE}/age_labels.npy')
gender_labels = np.load(f'{BASE}/gender_labels.npy')

# Split indexes
indices = np.arange(len(age_labels))
np.random.seed(42)
np.random.shuffle(indices)

split     = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx   = indices[split:]

# Labels split
age_train,    age_val    = age_labels[train_idx],    age_labels[val_idx]
gender_train, gender_val = gender_labels[train_idx], gender_labels[val_idx]

del age_labels, gender_labels
gc.collect()

# X mmap se load — RAM mein nahi jaayega
X = np.load(f'{BASE}/X.npy', mmap_mode='r')

# Generator — local indexes use karega
def data_generator(X, X_indices, age_labels, gender_labels, batch_size=32):
    local_age    = age_labels
    local_gender = gender_labels
    idx          = np.arange(len(X_indices))
    while True:
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch      = idx[i:i+batch_size]
            real_idx   = X_indices[batch]
            X_batch    = X[real_idx].copy()
            yield X_batch, {
                'age'   : local_age[batch],
                'gender': local_gender[batch]
            }

train_gen = data_generator(X, train_idx, age_train, gender_train)
val_gen   = data_generator(X, val_idx,   age_val,   gender_val)

train_steps = len(train_idx) // 32
val_steps   = len(val_idx)   // 32

# Model
input_layer = Input(shape=(128, 128, 3))
x = Conv2D(32,  (3,3), activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(64,  (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)

age_output    = Dense(1, name='age')(x)
gender_output = Dense(1, activation='sigmoid', name='gender')(x)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'age': 'mse', 'gender': 'binary_crossentropy'},
    metrics={'age': 'mae', 'gender': 'accuracy'}
)

model.summary()

# Callbacks
os.makedirs(f'{BASE}/models', exist_ok=True)

checkpoint = ModelCheckpoint(
    f'{BASE}/models/face_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)

print("Done!")