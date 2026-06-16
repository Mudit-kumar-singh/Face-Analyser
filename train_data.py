import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Data Loading
print("Loading data...")
X             = np.load("Processed_Data/X.npy")
age_labels    = np.load("Processed_Data/age_labels.npy")
gender_labels = np.load("Processed_Data/gender_labels.npy")
print(f"X shape      : {X.shape}")
print(f"Age labels   : {age_labels.shape}")
print(f"Gender labels: {gender_labels.shape}")

# Model Architecture
input_layer = Input(shape=(128, 128, 3))

x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
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
    loss={
        'age'   : 'mae',
        'gender': 'binary_crossentropy'
    },
    metrics={
        'age'   : 'mae',
        'gender': 'accuracy'
    }
)

model.summary()

# Train/Val Split
X_train, X_val, age_train, age_val, gender_train, gender_val = train_test_split(
    X, age_labels, gender_labels, test_size=0.2, random_state=42
)

del X
gc.collect()

# Callbacks
os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/face_model.h5",
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
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
# Train
model.fit(
    X_train,
    {'age': age_train, 'gender': gender_train},
    validation_data=(X_val, {'age': age_val, 'gender': gender_val}),
    epochs=1,
    batch_size=16,
    callbacks=[checkpoint, early_stop]
)

print("Model saved!")