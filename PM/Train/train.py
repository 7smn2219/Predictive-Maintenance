import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf  # Example using TensorFlow
import pickle

ok = "ok"
anomaly1 = "anomaly1"
anomaly2 = "anomaly2"

print("Loading files...")

# Load wav files from folders, extract labels from folder names
data, labels = [], []
for folder in [ok,anomaly1, anomaly2]:
    for filename in os.listdir(os.path.join(os.getcwd(),"/app/data/demo",folder)):
        print("in for loop with : ",filename)

        audio, sr = librosa.load(os.path.join("/app/data/demo/", folder,filename))

        data.append(audio)
        labels.append(folder)

# Convert labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


def extract_features(audio, sr=22050, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Average over time
    return mfccs

X_train = np.array([extract_features(x) for x in X_train])
X_test = np.array([extract_features(x) for x in X_test])

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer for 3 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Data Score {test_acc}")

import pickle
MODEL_PATH = '/app/model/model.pkl'
pickle.dump(model, open(MODEL_PATH, 'wb'))
