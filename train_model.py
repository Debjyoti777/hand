import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load preprocessed data
data_dict = np.load("landmarks_data.npy", allow_pickle=True).item()
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# 🔍 DIAGNOSTIC PRINTS — VERY IMPORTANT
print("Unique labels:", np.unique(labels))
print("Total samples:", len(labels))

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42
)

# Define improved neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(data.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(labels_categorical.shape[1], activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

print("\n🎯 Training complete!")
model.save("hand_gesture_model.keras")
np.save("label_encoder.npy", label_encoder.classes_)
print("📁 Model saved: hand_gesture_model.keras")
print("📁 Labels saved: label_encoder.npy")
