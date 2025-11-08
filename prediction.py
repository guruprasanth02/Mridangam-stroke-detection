import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('audio_classification_model.h5')

def extract_features(y, sr, n_mfcc=13, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]  # Truncate
    elif mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    return np.expand_dims(mfccs, axis=-1)

def predict_stroke(file_path, model, label_map):
    y_audio, sr = librosa.load(file_path, sr=None)
    mfccs = extract_features(y_audio, sr)
    mfccs = np.expand_dims(mfccs, axis=0)

    prediction = model.predict(mfccs)
    predicted_index = np.argmax(prediction, axis=1)[0]

    predicted_label = list(label_map.keys())[list(label_map.values()).index(predicted_index)]
    return predicted_label

audio_file_path = 'D:/stroke_detection/train/cha/224036__akshaylaya__cha-b-002.wav'

label_map = {'cha': 0, 'dheem': 1, 'thom': 2, 'tha': 3, 'ta': 4, 'thi': 5, 'num': 6, 'dhin': 7, 'tham': 8}

if __name__ == "__main__":
    predicted_stroke = predict_stroke(audio_file_path, model, label_map)
    print(f'Predicted Stroke: {predicted_stroke}')
