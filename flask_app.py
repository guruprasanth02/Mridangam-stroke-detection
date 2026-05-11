from flask import Flask, render_template, request, jsonify
import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import soundfile as sf
from python_speech_features import mfcc
import tensorflow as tf
from tensorflow import keras

from werkzeug.utils import secure_filename
from datetime import datetime
import io
# Initialize Flask app
app = Flask(__name__)

def safe_load_model(path):
    try:
        return keras.models.load_model(path)
    except Exception:
        # Fallback for older h5 models that use 'batch_shape' in InputLayer config
        try:
            import h5py

            with h5py.File(path, 'r') as f:
                # model config may be stored in attributes
                model_config = None
                if 'model_config' in f.attrs:
                    model_config = f.attrs['model_config']
                elif 'model_config' in f:
                    model_config = f['model_config'][()]

                if model_config is None:
                    raise ValueError('No model_config found in H5 file')

                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')

                # Replace legacy 'batch_shape' with 'batch_input_shape'
                fixed = model_config.replace('"batch_shape"', '"batch_input_shape"')

                # Provide known custom objects used in newer Keras/TensorFlow serializations
                try:
                    from keras.mixed_precision import policy as _policy
                    custom_objects = {'DTypePolicy': _policy.DTypePolicy}
                except Exception:
                    custom_objects = None

                if custom_objects:
                    with keras.utils.custom_object_scope(custom_objects):
                        model = keras.models.model_from_json(fixed)
                else:
                    model = keras.models.model_from_json(fixed)

                model.load_weights(path)
                return model
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")


# Load your pre-trained model
model = safe_load_model('mridangam_cnn_model.keras')

label_map = {'cha': 0, 'dheem': 1, 'thom': 2, 'tha': 3, 'ta': 4, 'thi': 5, 'num': 6, 'dhin': 7, 'tham': 8}

UPLOAD_FOLDER = 'uploads/'

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features(y, sr, n_mfcc=13, max_len=130):
    # Normalize audio
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    
    # Extract MFCC with larger FFT size to avoid truncation warning
    mfccs = mfcc(y, sr, numcep=n_mfcc, nfft=2048)
    mfccs = mfccs.T  # Transpose to (n_mfcc, time)
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]  # Truncate
    elif mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    return np.expand_dims(mfccs, axis=-1)


def predict_stroke(audio_buffer, model, label_map):
    y_audio, sr = sf.read(audio_buffer)
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
