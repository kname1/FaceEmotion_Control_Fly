import sys
sys.path.append('/usr/lib/python3/dist-packages')  # 换成你真实查到的路径

from picamera2 import Picamera2, Preview
import time
import threading
import collections
import cv2
import numpy as np
import tensorflow as tf

class CameraThread:
    def __init__(self):
        self.running_flag = True
        self.interpreter = self.get_model()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.types = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.last_emotion = None
        self.is_judging = False
        self.judge_lock = threading.Lock()
        self.emotion_window = collections.deque(maxlen=4)
        self.last_detection_time = time.time()
        self.sad_start_time = None
        self.drone_started = False
        self.drone_stopped_due_to_sadness = False
        
        # 初始化 Picamera2
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_still_configuration(
            main={"size": (640, 480)},
            lores={"size": (640, 480)},
            display="lores"
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

    def get_model(self):
        interpreter = tf.lite.Interpreter(model_path="module.tflite")
        interpreter.allocate_tensors()
        return interpreter

    def showCamera(self):
        while self.running_flag:
            frame = self.picam2.capture_array()
            if frame is None:
                print("无法获取摄像头画面")
                break
            self.getDetectCamera(frame)
            time.sleep(1/30)

    def getDetectCamera(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(150, 150))

        for (x, y, w, h) in faceRects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame[y:y + h, x:x + w]
            self.predict_emotion(face_roi)

    def predict_emotion(self, face_roi):
        img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        type_index = np.argmax(output_data)
        emotion_label = self.types[type_index]

        current_time = time.time()
        if current_time - self.last_detection_time >= 0.5:
            self.check_emotion_change(emotion_label)
            self.last_detection_time = current_time

    def check_emotion_change(self, new_emotion):
        self.emotion_window.append(new_emotion)
        if len(self.emotion_window) == self.emotion_window.maxlen:
            threading.Thread(target=self.judge_emotion_change, args=(new_emotion,)).start()

    def judge_emotion_change(self, new_emotion):
        with self.judge_lock:
            if self.is_judging:
                return
            self.is_judging = True
            counts = collections.Counter(self.emotion_window)
            most_common_emotion, count = counts.most_common(1)[0]
            if count >= 3:
                self.execute_drone_control(most_common_emotion)
            self.is_judging = False

    def execute_drone_control(self, emotion):
        if not self.drone_started:
            if emotion == "Happy":
                print("无人机启动")
                self.drone_started = True
                self.drone_stopped_due_to_sadness = False
            return

        if self.drone_stopped_due_to_sadness:
            if emotion == "Happy":
                print("无人机启动")
                self.drone_started = True
                self.drone_stopped_due_to_sadness = False
            return

        if emotion == "Happy":
            print("无人机上升")
        elif emotion == "Sad":
            if self.sad_start_time is None:
                self.sad_start_time = time.time()
            elif time.time() - self.sad_start_time >= 8:
                print("由于长时间的悲伤，无人机关闭")
                self.drone_started = False
                self.drone_stopped_due_to_sadness = True
                self.sad_start_time = None
            else:
                print("无人机下降")
        else:
            self.sad_start_time = None

        if emotion == "Surprise" and self.drone_started:
            print("无人机前进")
        elif emotion == "Angry" and self.drone_started:
            print("无人机后退")

    def stopRunning(self):
        self.running_flag = False
        self.picam2.stop()

    def run(self):
        self.showCamera()

if __name__ == '__main__':
    camera_thread = CameraThread()
    try:
        camera_thread.run()
    except KeyboardInterrupt:
        camera_thread.stopRunning()
