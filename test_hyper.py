import cv2
import dlib
import numpy as np
import joblib
import os

# Đường dẫn các mô hình
HAARCASCADE_PATH = "data/haarcascade_frontalface_alt2.xml"
LBP_PATH = "data/lbpcascade_frontalface.xml"
LANDMARK_MODEL_PATH = "data/shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "face_shape_model_svr_svc.pkl"

# Nhãn khuôn mặt
FACE_SHAPE_LABELS = {
    0.0: "Heart",
    1.0: "Oblong",
    2.0: "Oval",
    3.0: "Round",
    4.0: "Square"
}

# Kiểm tra file mô hình đã tồn tại chưa
for path in [HAARCASCADE_PATH, LBP_PATH, LANDMARK_MODEL_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Không tìm thấy {path}")

# Load mô hình nhận diện khuôn mặt
detector_dlib = dlib.get_frontal_face_detector()
detector_haar = cv2.CascadeClassifier(HAARCASCADE_PATH)
detector_lbp = cv2.CascadeClassifier(LBP_PATH)
landmark_predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

# Load mô hình đã huấn luyện
svr, svc = joblib.load(MODEL_PATH)
print(" Mô hình đã được tải thành công.")

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

# Hàm phát hiện khuôn mặt
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1️ Dlib
    faces = detector_dlib(gray)
    if len(faces) > 0:
        return faces[0]

    # 2️ Haar Cascade
    faces = detector_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h)

    # 3️ LBP Cascade
    faces = detector_lbp.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return dlib.rectangle(x, y, x + w, y + h)

    return None

# Hàm dự đoán hình dạng khuôn mặt
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
    shape_label = svc.predict([features])[0]  # Dự đoán nhãn bằng SVC
    face_shape = FACE_SHAPE_LABELS.get(shape_label, "Unknown")

    print(f"🎯 Kết quả: {face_shape}")

    # Hiển thị ảnh với nhãn
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, face_shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Predicted Face Shape", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  Thử nghiệm với một ảnh
if __name__ == "__main__":
    image_path = "data/heart.jpg"  # Đổi thành đường dẫn ảnh của bạn
    predict_face_shape(image_path)
