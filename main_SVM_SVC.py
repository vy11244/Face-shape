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

# Định nghĩa nhãn khuôn mặt cho từng loại
FACE_SHAPE_LABELS = {
    "Heart": 0,
    "Oblong": 1,
    "Oval": 2,
    "Round": 3,
    "Square": 4
}
REVERSE_LABELS = {v: k for k, v in FACE_SHAPE_LABELS.items()}

DATASET_DIR = "dataset/training_set"
TEST_DIR = "dataset/testing_set"
MODEL_PATH = "face_shape_model.pkl"

# Hàm trích xuất đặc trưng từ landmarks
def extract_features(landmarks):
    jaw_width = np.linalg.norm(landmarks[0] - landmarks[16])
    cheekbone_width = np.linalg.norm(landmarks[3] - landmarks[13])
    face_length = np.linalg.norm(landmarks[8] - (landmarks[0] + landmarks[16]) / 2)
    forehead_width = np.linalg.norm(landmarks[19] - landmarks[24])

    return [
        jaw_width / face_length,
        cheekbone_width / face_length,
        forehead_width / face_length,
        jaw_width / cheekbone_width,
        cheekbone_width / forehead_width,
        forehead_width / jaw_width
    ]

# Hàm phát hiện khuôn mặt (tích hợp 3 phương pháp)
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1️ Thử phát hiện bằng Dlib
    faces = detector_dlib(gray)
    if len(faces) > 0:
        return faces[0]

    # 2️ Thử phát hiện bằng Haar Cascade
    faces = detector_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h)

    # 3️ Thử phát hiện bằng LBP Cascade
    faces = detector_lbp.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h)

    return None

# Hàm load dataset từ thư mục
def load_dataset(data_dir):
    X_train = []
    y_train = []

    print(f" Đang tải dataset từ {data_dir}...")
    for face_shape, label in FACE_SHAPE_LABELS.items():
        folder_path = os.path.join(data_dir, face_shape)
        if not os.path.exists(folder_path):
            print(f" Không tìm thấy thư mục {face_shape}, bỏ qua...")
            continue

        for img_name in os.listdir(folder_path):
            if img_name.lower() == "desktop.ini":
                continue
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f" Không thể đọc ảnh {img_name}, bỏ qua...")
                continue

            face_rect = detect_face(image)
            if face_rect is None:
                print(f" Không tìm thấy khuôn mặt trong ảnh {img_name}, bỏ qua...")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape = landmark_predictor(gray, face_rect)
            landmarks = np.array([(p.x, p.y) for p in shape.parts()])

            features = extract_features(landmarks)
            X_train.append(features)
            y_train.append(label)

            print(f" Đã xử lý {len(X_train)} ảnh: {img_name} ({face_shape})")

    return np.array(X_train), np.array(y_train)

# Huấn luyện mô hình
def train_model():
    X_train, y_train = load_dataset(DATASET_DIR)

    if X_train.shape[0] == 0:
        raise ValueError(" Không có dữ liệu ảnh để huấn luyện!")

    print(f" Tổng số ảnh huấn luyện: {X_train.shape[0]}")

    # Kết hợp dữ liệu từ modelData.py (nếu có)
    if len(X) > 0:
        print(" Đang kết hợp dữ liệu modelData.py...")
        X_train = np.vstack([X, X_train])
        y_train = np.hstack([np.zeros(len(X)), y_train])  # Gán nhãn giả cho dữ liệu từ modelData.py

    print(" Bắt đầu huấn luyện mô hình SVM...")
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X_train, y_train)
    print(" Huấn luyện xong, đang lưu mô hình...")

    joblib.dump(svm_model, MODEL_PATH)
    print(" Mô hình đã được huấn luyện và lưu lại.")

    return svm_model

# Đánh giá mô hình với dữ liệu testing_set
def evaluate_model(model):
    X_test, y_test = load_dataset(TEST_DIR)

    if X_test.shape[0] == 0:
        print("⚠ Không có dữ liệu để đánh giá!")
        return

    accuracy = model.score(X_test, y_test)
    print(f" Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

# Tải hoặc huấn luyện mô hình
try:
    svm_model = joblib.load(MODEL_PATH)
    print(" Mô hình đã được tải thành công.")
except FileNotFoundError:
    print(" Không tìm thấy mô hình, tiến hành huấn luyện...")
    svm_model = train_model()

# Chạy thử nghiệm trên ảnh
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

    features = extract_features(landmarks)
    shape_label = svm_model.predict([features])[0]
    face_shape = REVERSE_LABELS.get(shape_label, "Unknown")

    print(f" Kết quả: {face_shape}")

# Chạy thử nghiệm
if __name__ == "__main__":
    evaluate_model(svm_model)
