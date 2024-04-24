import sys
import cv2
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import mediapipe as mp
import time
import pyttsx3

class HandGestureRecognitionApp(QMainWindow):
    def __init__(self):
        super(HandGestureRecognitionApp, self).__init__()

        self.setWindowTitle("Hand Gesture Recognition")
        self.setGeometry(100, 100, 1000, 600)

        self.init_ui()

        self.model_dict = pickle.load(open("model.p", 'rb'))
        self.model = self.model_dict['model']
        self.labels_dict = {0: '  ', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q' , 18: 'R' , 19: 'S' , 20: 'T' , 21: 'U' , 22: 'V' , 23: 'W' , 24: 'X' , 25: 'Y' , 26: 'Z'}

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        self.last_prediction_time = time.time()
        self.predicted_alphabets = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_camera_feed)

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.camera_label = QLabel()
        self.text_label = QLabel()
        self.text_label.setStyleSheet("font-size: 20pt;")
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_camera_feed)
        speech_button = QPushButton("Speech")
        speech_button.clicked.connect(self.convert_to_speech)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.text_label)
        layout.addWidget(start_button)
        layout.addWidget(speech_button)
#function for camera feed
    def process_camera_feed(self):
        ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        current_time = time.time()
        time_difference = current_time - self.last_prediction_time
        if results.multi_hand_landmarks and time_difference >= 5:
            self.last_prediction_time = current_time
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) - 20
            y2 = int(max(y_) * H) - 20
            prediction = self.model.predict([np.asarray(data_aux)])
            predicted_character = self.labels_dict[int(prediction[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            self.text_label.setText(self.text_label.text() + predicted_character)
            self.predicted_alphabets.append(predicted_character)
            concatenated_text = "".join(self.predicted_alphabets)
            print("Predicted alphabets:", self.predicted_alphabets)
            print("Concatenated text:", concatenated_text)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap)
#function for start camera feed
    def start_camera_feed(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(10)
#function for convert to speech
    def convert_to_speech(self):
        concatenated_text = "".join(self.predicted_alphabets)
        engine = pyttsx3.init()
        engine.setProperty('rate', 100)
        engine.say(concatenated_text)
        engine.runAndWait()
#function to start and close of application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = HandGestureRecognitionApp()
    mainWin.show()
    sys.exit(app.exec_())
