import os
import numpy as np
import librosa
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATASET_PATH = "D:\\Mridangam-stroke-detection\\train"   # <-- CHANGE THIS
N_MFCC = 13
MAX_LEN = 130
TEST_SIZE = 0.25
VAL_SIZE = 0.2
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 32

# =========================
# FEATURE EXTRACTION
# =========================
def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

# =========================
# LOAD DATASET (NO JSON)
# =========================
def load_dataset(dataset_path):
    X, y = [], []
    label_map = {}
    label_index = 0

    for stroke in sorted(os.listdir(dataset_path)):
        stroke_path = os.path.join(dataset_path, stroke)
        if not os.path.isdir(stroke_path):
            continue

        label_map[label_index] = stroke

        for file in os.listdir(stroke_path):
            if file.endswith(".wav"):
                mfcc = extract_mfcc(os.path.join(stroke_path, file))
                X.append(mfcc)
                y.append(label_index)

        label_index += 1

    X = np.array(X)
    y = np.array(y)

    return X, y, label_map

# =========================
# PREPARE DATASETS
# =========================
def prepare_datasets():
    X, y, label_map = load_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train
    )

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test, label_map

# =========================
# MODEL ARCHITECTURE
# =========================
def build_model(input_shape, num_classes):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu',
        input_shape=input_shape,
        kernel_regularizer=keras.regularizers.l2(0.001)
    ))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001)
    ))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(
        32, (2, 2), activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001)
    ))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        64, activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001)
    ))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model

# =========================
# TRAIN & EVALUATE
# =========================
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train", "Validation"])

    plt.show()

if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test, label_map = prepare_datasets()

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(label_map)

    model = build_model(input_shape, num_classes)

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop]
    )

    model.save("mridangam_cnn_model.keras")

    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    plot_history(history)
