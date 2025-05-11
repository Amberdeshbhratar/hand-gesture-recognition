# Hand Gesture Recognition

**This project provides a real-time hand gesture recognition system using a webcam, TensorFlow, MediaPipe, and a Flask web server. It detects hand gestures and classifies them into predefined actions using a trained deep learning model.


## 🧠 Features

- Real-time hand gesture recognition via webcam.
- Trained on hand landmarks extracted using MediaPipe Holistic.
- Simple Flask web interface to visualize predictions.
- Toggleable recording for efficient inference.

## 📂 Project Structure

├── app.py                          # Flask application for live inference<br>
├── action2.h5                      # Trained Keras model for gesture classification<br>
├── HandGestureRealTime_Train.ipynb# Notebook for training the model<br>
├── HandGestureRealTime_Test.ipynb # Notebook for testing the model<br>
├── templates/<br>
│          └── index.html                  # Web interface to view predictions <br>


## 🚀 Getting Started
### Install Dependencies
```bash
pip install opencv-python
pip install numpy
pip install mediapipe
pip install tensorflow
pip install flask
```
## 🚀 Usage
### 1. Run the Flask App
```bash
python app.py
```
Access the web interface at http://localhost:3000.
### 2. Toggle Gesture Recording
Use the buttons on the frontend (index.html) to start and stop recording. The system captures 30 frames and then predicts the gesture.

### 3. API Endpoints
- /video_feed: Live video stream with landmarks and status overlay.
- /toggle_recording: POST endpoint to start/stop recording.
- /result: Returns the most recent gesture prediction.
  
## 📓 Notebooks
HandGestureRealTime_Train.ipynb: Used to collect and train gesture data into action2.h5.
HandGestureRealTime_Test.ipynb: For testing the model on collected sequences.

## 🧪 Model Details
Model Type: Sequential (loaded from action2.h5)
Input: 30-frame sequences of hand landmarks (left + right hands)
Output: Softmax prediction over predefined classes
