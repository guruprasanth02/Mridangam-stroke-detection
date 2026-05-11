import gradio as gr
import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import soundfile as sf
from python_speech_features import mfcc
import tensorflow as tf
from tensorflow import keras

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

def gradio_predict(audio_file):
    if audio_file is None:
        return "No audio file provided."
    try:
        with open(audio_file, "rb") as f:
            predicted_stroke = predict_stroke(f, model, label_map)
        return f"Predicted Stroke: {predicted_stroke}"
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Mridangam Stroke Detection",
    description="Upload an audio file (wav, mp3, etc.) to predict the Mridangam stroke type."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)

