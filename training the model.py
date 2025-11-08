import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_audio_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load the audio file with its original sampling rate
        return y, sr
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def extract_features(y, sr, n_mfcc=13, max_len=100):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] > max_len:
            mfccs = mfccs[:, :max_len]  # Truncate
        elif mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((n_mfcc, max_len))

def prepare_dataset(directory, n_mfcc=13, max_len=100):
    X = []
    y = []
    label_map = {}
    current_label = 0

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue

        if folder not in label_map:
            label_map[folder] = current_label
            current_label += 1

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                y_audio, sr = load_audio_file(file_path)
                if y_audio is not None:
                    mfccs = extract_features(y_audio, sr, n_mfcc, max_len)
                    mfccs = np.expand_dims(mfccs, axis=-1)
                    X.append(mfccs)
                    y.append(label_map[folder])

    X = np.array(X)
    y = np.array(y)

    return X, y, label_map

dataset_path = 'D:/stroke_detection/train'

X, y, label_map = prepare_dataset(dataset_path)

print(f"Dataset loaded with {len(X)} samples.")

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(label_map)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = build_model(input_shape, num_classes)

history = model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('audio_classification_model.h5')
