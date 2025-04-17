import cv2
import dlib
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from data.modelData import X  # Dữ liệu đặc trưng từ modelData.py

# Định nghĩa đường dẫn cho các bộ nhận diện khuôn mặt
HAARCASCADE_PATH = "data/haarcascade_frontalface_alt2.xml"
LBP_PATH = "data/lbpcascade_frontalface.xml"
LANDMARK_MODEL_PATH = "data/shape_predictor_68_face_landmarks.dat"

# Kiểm tra sự tồn tại của các file mô hình
for path in [HAARCASCADE_PATH, LBP_PATH, LANDMARK_MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f" Không tìm thấy {path}")

# Load các mô hình nhận diện khuôn mặt
detector_dlib = dlib.get_frontal_face_detector()
detector_haar = cv2.CascadeClassifier(HAARCASCADE_PATH)
detector_lbp = cv2.CascadeClassifier(LBP_PATH)
landmark_predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

# Định nghĩa nhãn khuôn mặt
FACE_SHAPE_LABELS = { "Heart": 0, "Oblong": 1, "Oval": 2, "Round": 3, "Square": 4 }
REVERSE_LABELS = {v: k for k, v in FACE_SHAPE_LABELS.items()}

MODEL_PATH = "face_shape_model.pkl"

# Hàm trích xuất đặc trưng từ landmarks
def extract_features(landmarks):
    jaw_width = np.linalg.norm(landmarks[0] - landmarks[16])  # Độ rộng xương hàm
    cheekbone_width = np.linalg.norm(landmarks[3] - landmarks[13])  # Độ rộng xương gò má
    face_length = np.linalg.norm(landmarks[8] - (landmarks[0] + landmarks[16]) / 2)  # Chiều dài khuôn mặt
    forehead_width = np.linalg.norm(landmarks[19] - landmarks[24])  # Độ rộng trán

    features = [
        jaw_width / face_length,
        cheekbone_width / face_length,
        forehead_width / face_length,
        jaw_width / cheekbone_width,
        cheekbone_width / forehead_width,
        forehead_width / jaw_width
    ]

    # 🛠️ Thêm dòng in ra tỷ lệ đặc trưng
    print(f"Tỷ lệ đặc trưng khuôn mặt: {features}")

    return features

# Hàm phát hiện khuôn mặt (dùng 3 phương pháp)
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dlib
    faces = detector_dlib(gray)
    if len(faces) > 0:
        return faces[0]

    # Haar Cascade
    faces = detector_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h)

    # LBP Cascade
    faces = detector_lbp.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h)

    return None

# Tải hoặc huấn luyện mô hình
try:
    svm_model = joblib.load(MODEL_PATH)
    print(" Mô hình đã được tải thành công.")
except FileNotFoundError:
    print(" Không tìm thấy mô hình, hãy huấn luyện trước!")

# Hàm chạy thử trên một ảnh
def predict_face_shape(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f" Không tìm thấy ảnh {image_path}")
        return

    face_rect = detect_face(image)
    if face_rect is None:
        print(" Không tìm thấy khuôn mặt!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = landmark_predictor(gray, face_rect)
    landmarks = np.array([(p.x, p.y) for p in shape.parts()])

    # Trích xuất đặc trưng khuôn mặt
    features = extract_features(landmarks)

    # Kiểm tra mô hình đã load thành công chưa
    if svm_model is None:
        print(" Mô hình chưa được tải, không thể dự đoán.")
        return

    # Dự đoán hình dạng khuôn mặt
    shape_label = svm_model.predict([features])[0]
    face_shape = REVERSE_LABELS.get(shape_label, "Unknown")

    # Hiển thị ảnh với kết quả dự đoán
    x, y, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, face_shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f" Kết quả dự đoán: {face_shape}")

# Chạy thử nghiệm
if __name__ == "__main__":
    test_image_path = r"data/oval.jpg"  
    predict_face_shape(test_image_path)
