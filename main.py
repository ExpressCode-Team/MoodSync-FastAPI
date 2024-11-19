from fastapi import FastAPI, File, UploadFile
import dlib
import mediapipe as mp
import numpy as np
import pickle
import cv2
import os
import shutil
from math import acos, degrees

app = FastAPI()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
label = ['angry', 'happy', 'neutral', 'sad']

with open('file/model.pkl', 'rb') as file:
    modelml = pickle.load(file)
    
with open('file/modeldl.pkl', 'rb') as file:
    modeldl = pickle.load(file)

with open('file/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


def preprocess_img(img, size=(128, 128)):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img

# Fungsi untuk menghitung jarak antar dua titik
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Fungsi untuk menghitung sudut antara tiga titik
def calculate_angle(p1, p2, p3):
    # Menghitung vektor
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Menghitung produk titik dan besar vektor
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Menghitung sudut dalam derajat
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = acos(np.clip(cos_theta, -1.0, 1.0))  # Menangani kemungkinan kesalahan numerik
    return degrees(angle)

# Fungsi untuk mengekstrak fitur jarak, sudut, dan segitiga
def extract_distance_angle_triangle_features(landmarks):
    features = []

    triangle_sides = [
        (9, 40),
        (40, 31),
        (31, 9),
        (0, 40),
        (40, 37),
        (37, 0),
        (4, 11),
        (11, 15),
        (15, 4),
        (5, 18),
        (18, 20),
        (20, 5),
    ]
    # Fitur jarak antar landmark
    for i, j in triangle_sides:
        distance = calculate_distance(landmarks[i], landmarks[j])
        features.append(distance)

    # Fitur sudut antar tiga titik landmark
    triangle_indices = [
        (9, 40, 31),  # alis ke mulut (kanan dari kita)
        (0, 40, 37),  # alis ke mulut (kiri dari kita)
        (4, 11, 15),  # alis ke mata (kiri dari kita)
        (5, 18, 20),  # alis ke mata (kanan dari kita)
            ]

    for i, j, k in triangle_indices:
        angle1 = calculate_angle(landmarks[i], landmarks[j], landmarks[k])
        angle2 = calculate_angle(landmarks[j], landmarks[k], landmarks[i])
        angle3 = calculate_angle(landmarks[k], landmarks[i], landmarks[j]) 
        features.append(angle1)
        features.append(angle2)
        features.append(angle3)

    return features

def detect_relevant_landmarks(img):
    # Indeks landmark untuk setiap bagian wajah
    feature_landmarks = {
        "left_eyebrow": list(range(17, 22)),   # Alis kiri
        "right_eyebrow": list(range(22, 27)),  # Alis kanan
        "left_eye": list(range(36, 42)),       # Mata kiri
        "right_eye": list(range(42, 48)),      # Mata kanan
        "nose": list(range(27, 36)),           # Hidung
        "mouth": list(range(48, 68)),          # Mulut
    }

    # Mengubah gambar ke grayscale
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Deteksi wajah
    faces = detector(img_gray)
    processed_img = []
    features_list = []

    for face in faces:
            landmarks = predictor(img_gray, face)
            relevant_landmarks = []

            # Ambil landmark berdasarkan fitur yang relevan
            for feature, indices in feature_landmarks.items():
                relevant_landmarks.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in indices])

            # Ekstraksi fitur jarak, sudut, dan segitiga
            features_list.append(extract_distance_angle_triangle_features(relevant_landmarks))

    return features_list if features_list else None

def standardize_features(features):
    standardized_features = scaler.transform(features)
    return standardized_features

def prepare_data(img, target_size=(128, 128)):
    image = np.array(img, dtype=np.float32).reshape(-1, target_size[0], target_size[1], 1)
    image = image / 255.0 
    return image

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/ml/")
async def predict_image(file: UploadFile = File(...)):
    # Cek apakah folder 'tmp' ada
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Simpan file sementara
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Baca gambar
    img = cv2.imread(file_location)
    if img is None:
        os.remove(file_location)
        return {"label": "Failed to read image"}

    # Preprocess gambar dan deteksi landmark
    img_preprocessed = preprocess_img(img)
    
    if img_preprocessed is None:
        os.remove(file_location)
        return {"label": "Image cannot be preprocessed"}
    
    features_list = detect_relevant_landmarks(img_preprocessed)
    if features_list is None:
        os.remove(file_location)
        return {"label": "Relevant landmark not detected"}

    # Standarisasi fitur
    standardized_features = standardize_features(features_list)
    if standardized_features is None:
        os.remove(file_location)
        return {"label": "Failed to standardize features"}
    

    # Prediksi dengan model
    predict = modelml.predict(standardized_features)

    # Hapus file sementara
    os.remove(file_location)

    if isinstance(predict, np.ndarray):
        predict = predict.tolist()

    return {"predict": str(predict), "label": label[predict[0]]}

@app.post("/predict/dl/")
async def predict_image(file: UploadFile = File(...)):
    # Cek apakah folder 'tmp' ada
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Simpan file sementara
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Baca gambar
    img = cv2.imread(file_location)
    if img is None:
        os.remove(file_location)
        return {"label": "Failed to read image"}

    img_preprocessed = preprocess_img(img)
    print(img_preprocessed.shape)
    if img_preprocessed is None:
        os.remove(file_location)
        return {"label": "Failed to preprocess image"}
    
    img_prepare = prepare_data(img_preprocessed)
    
    predict = modeldl.predict(img_prepare)

    os.remove(file_location)

    if isinstance(predict, np.ndarray):
        predict = predict.tolist()
        
    return {"predict": str(predict), "label": label[np.argmax(predict)]}

@app.post("/predict/mlwithtmp/")
async def predict_image(file: UploadFile = File(...)):
    # Cek apakah folder 'tmp' ada
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Simpan file sementara
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Baca gambar
    img = cv2.imread(file_location)
    if img is None:
        return {"label": "Failed to read image"}

    # Preprocess gambar dan deteksi landmark
    img_preprocessed = preprocess_img(img)
    
    if img_preprocessed is None:
        return {"label": "Image cannot be preprocessed"}
    
    features_list = detect_relevant_landmarks(img_preprocessed)
    if features_list is None:
        return {"label": "Relevant landmark not detected"}

    # Standarisasi fitur
    standardized_features = standardize_features(features_list)
    if standardized_features is None:
        return {"label": "Failed to standardize features"}

    # Prediksi dengan model
    predict = modelml.predict(standardized_features)

    if isinstance(predict, np.ndarray):
        predict = predict.tolist()

    return {"predict": str(predict), "label": label[predict[0]]}
