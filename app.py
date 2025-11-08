from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

from werkzeug.utils import secure_filename
from datetime import datetime
import io
# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model = load_model('audio_classification_model.h5')

label_map = {'cha': 0, 'dheem': 1, 'thom': 2, 'tha': 3, 'ta': 4, 'thi': 5, 'num': 6, 'dhin': 7, 'tham': 8}

UPLOAD_FOLDER = 'uploads/'

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features(y, sr, n_mfcc=13, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]  # Truncate
    elif mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    return np.expand_dims(mfccs, axis=-1)


def predict_stroke(audio_buffer, model, label_map):
    y_audio, sr = librosa.load(audio_buffer, sr=None)
    mfccs = extract_features(y_audio, sr)
    mfccs = np.expand_dims(mfccs, axis=0)

    prediction = model.predict(mfccs)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = list(label_map.keys())[list(label_map.values()).index(predicted_index)]
    return predicted_label


# Define a route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Define a route to handle file upload and stroke prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an audio file
    allowed_extensions = {'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'}
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_extension not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an audio file.'})

    # --- Create a unique filename to prevent overwrites ---
    # 1. Sanitize the filename to remove insecure characters
    original_filename = secure_filename(file.filename)
    # 2. Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 3. Create the new unique filename
    unique_filename = f"{timestamp}_{original_filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        # Use a 'with' statement to ensure the file is closed after saving
        with open(file_path, "wb") as f:
            f.write(file.read())

        # Now that the file is saved and closed, open it again for prediction
        with open(file_path, "rb") as f:
            # Perform stroke prediction from the file
            predicted_stroke = predict_stroke(f, model, label_map)

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

    return jsonify({'predicted_stroke': predicted_stroke})


if __name__ == "__main__":
    app.run(debug=True)
