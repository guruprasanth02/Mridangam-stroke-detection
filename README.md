# 🎵 Mridangam Stroke Detection

## 🌟 Overview

This project leverages a Convolutional Neural Network (CNN) to classify various strokes (syllables) of the Mridangam, a traditional South Indian percussion instrument, from audio files. It provides both training scripts for model development and a user-friendly Flask web application for real-time stroke prediction, making it accessible for musicians, researchers, and enthusiasts. 🎶

## ✨ Features

- **🎧 Audio Classification**: Accurately identifies Mridangam strokes such as Cha, Dheem, Dhin, Num, Ta, Tha, Tham, Thi, and Thom using MFCC features.
- **🌐 Web Interface**: Intuitive Flask-based web app for uploading audio files and receiving instant predictions.
- **🤖 Model Training**: Customizable training script to build and train the CNN model on your dataset.
- **⚡ Real-time Prediction**: Supports single-file predictions via script or web upload.
- **📁 Dataset Organization**: Structured dataset handling for easy integration of new audio samples.

## ⚙️ Setup and Installation

Follow these steps to set up the project environment. 🚀

1. **📥 Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd Mridangam-stroke-detection
   ```

2. **🐍 Create a Virtual Environment**
   It is highly recommended to use a virtual environment to manage project dependencies.
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **📦 Install Dependencies**
   Install all the required libraries using the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Model Details

The CNN model is designed for audio classification using Mel-Frequency Cepstral Coefficients (MFCCs) as input features. The architecture includes convolutional layers for feature extraction, followed by pooling and dense layers for classification. The model is trained on a dataset of Mridangam strokes and achieves high accuracy in identifying different syllables. 📈

For more details on the model architecture, refer to `training the model.py`.

## ▶️ Usage

### 1. 📊 Dataset Preparation

Before training, organize your audio files in a directory structure where each sub-directory is named after a stroke and contains the corresponding `.wav` files. 🎼

```
<dataset_path>/
├── cha/
│   ├── audio1.wav
│   └── audio2.wav
├── dheem/
│   ├── audio3.wav
│   └── ...
└── ...
```

Update the `dataset_path` variable in `training the model.py` to point to your dataset directory.

### 2. 🤖 Training the Model

Run the training script to train the model on your dataset. 🏋️‍♂️
```bash
python "training the model.py"
```
This generates the `audio_classification_model.h5` file in the project root.

### 3. 🌐 Running the Web Application for Prediction

Start the Flask application for web-based predictions: 🌍
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` in your browser to upload audio files and receive predictions.

### 4. 🔍 Single-File Prediction

Use the prediction script for individual audio files: 🔬
```bash
python prediction.py
```
Ensure the audio file path is specified in the script.

## 🤝 Contributing

Contributions are welcome! Please follow these steps: 🙌

1. Fork the repository. 🍴
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request. 📤

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. ⚖️
