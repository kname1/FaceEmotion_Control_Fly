import time
import threading
import collections
import cv2
import numpy as np
from tensorflow.python.keras.saving.save import load_model

class CameraThread:
    def __init__(self):
        self.running_flag = True
        self.model = self.get_model()
        self.types = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.last_emotion = None
        self.is_judging = False
        self.judge_lock = threading.Lock()
        self.emotion_window = collections.deque(maxlen=4)
        self.last_detection_time = time.time()
        self.sad_start_time = None
        self.drone_started = False
        self.drone_stopped_due_to_sadness = False
        # 初始化摄像头（只打开一次）
        self.cap = cv2.VideoCapture(0)

    def get_model(self):
        # 加载预训练模型，确保 module.h5 文件存在
        face_model = load_model("module.h5")
        return face_model

    def showCamera(self):
        while self.running_flag:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
            self.getDetectCamera(frame)
            # 控制帧率，约 60 FPS
            time.sleep(1/60)

    def getDetectCamera(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(150, 150))

        # 对检测到的人脸画矩形，并调用情绪预测
        for (x, y, w, h) in faceRects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            self.predict_emotion(face_roi)

        # 在 OpenCV 窗口中显示结果
        cv2.imshow("Camera", frame)
        # 检测键盘输入：按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stopRunning()

    def predict_emotion(self, face_roi):
        # 将人脸区域预处理为模型需要的输入格式
        img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape((1, 224, 224, 3))

        pred = self.model.predict(img)
        type_index = np.argmax(pred)
        emotion_label = self.types[type_index]
        # print("检测到的表情:", emotion_label)

        # 每0.5秒更新一次情绪窗口并判断
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
            if count >= 3:  # 如果75%的时间都是这个表情
                self.execute_drone_control(most_common_emotion)
            self.is_judging = False

    def execute_drone_control(self, emotion):
        # 根据当前无人机状态和检测到的情绪执行控制逻辑
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
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.showCamera()

if __name__ == '__main__':
    camera_thread = CameraThread()
    try:
        camera_thread.run()
    except KeyboardInterrupt:
        camera_thread.stopRunning()
