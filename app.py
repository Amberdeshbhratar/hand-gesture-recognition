import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from flask import Flask, render_template, Response, jsonify, request
import threading

app = Flask(__name__)

actions = np.array(['Hello', 'All The Best', 'Peace', 'Call me', 'Nice'])
model = Sequential()
model.add(load_model('actions.h5'))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

recording = False
latest_result = ""
lock = threading.Lock()

def mediapipe_detection(image, model_mp):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model_mp.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def gen_frames():
    global latest_result, recording
    sequence = []
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image, results_mp = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results_mp)

            if recording:
                keypoints = extract_keypoints(results_mp)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    pred = actions[np.argmax(res)]
                    with lock:
                        latest_result = pred
            # else: do not update latest_result

            # Draw recording status
            status = "Recording" if recording else "Stopped"
            color = (0,255,0) if recording else (0,0,255)
            cv2.rectangle(image, (0,0), (200, 40), color, -1)
            cv2.putText(image, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global recording
    data = request.get_json()
    command = data.get('command')
    if command == 'start':
        recording = True
    elif command == 'stop':
        recording = False
    return jsonify({'status': 'ok', 'recording': recording})

@app.route('/result')
def get_result():
    with lock:
        result = latest_result
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run()
