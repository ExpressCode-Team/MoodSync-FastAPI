from fastapi import FastAPI, File, UploadFile
import dlib
import mediapipe as mp
import numpy as np
import pickle
import cv2
import os
import shutil

app = FastAPI()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
label = ['angry', 'happy', 'neutral', 'sad']

with open('model.pkl', 'rb') as file:
    modelml = pickle.load(file)
    
with open('modeldl.pkl', 'rb') as file:
    modeldl = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('pca.pkl', 'rb') as file:
    pca = pickle.load(file) 

def preprocess_img(img, size=(128, 128)):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img

def extract_features(landmarks):
    landmarks = np.array(landmarks)
    return np.array([np.linalg.norm(landmarks[i] - landmarks[j]) for i in range(len(landmarks)) for j in range(i + 1, len(landmarks))])

# def detect_relevant_landmarks(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
#     results = mp_face_mesh.process(img)
#     features_list = []

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = img.shape
#             relevant_landmarks = [
#                 (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark)
#                 if i in range(17, 22) or i in range(22, 27) or i in range(36, 42) or i in range(42, 48) or i in range(48, 68)
#             ]
#             features = extract_features(relevant_landmarks)
#             features_list.append(features)
#     return features_list

def detect_relevant_landmarks(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    faces = detector(img_gray)
    relevant_landmarks = []
    features_list = []

    for face in faces:
        landmarks = predictor(img_gray, face)
        relevant_landmarks = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(68) 
            if i in range(18, 22) or
               i in range(23, 27) or 
               i in range(37, 42) or    
               i in range(43, 48) or
               i in range(61, 68) or
               i in range(49, 60) or
               i in range(32, 36)
        ]
        
        if not relevant_landmarks:
            continue

        # Gambar titik landmark untuk visualisasi
        for x, y in relevant_landmarks:
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        features = extract_features(relevant_landmarks)
        features_list.append(features)

    return features_list if features_list else None

def standardize_features(features):
    standardized_features = scaler.transform(features)
    return standardized_features


def pca_features(features):
    pca_features = pca.transform(features)
    return pca_features

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

    # # Periksa tipe file
    # if file.content_type.split('/')[0] != 'image':
    #     return {"label": "Invalid file type"}

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
    
    pca_transformed_features = pca_features(standardized_features)

    # Prediksi dengan model
    predict = modelml.predict(pca_transformed_features)

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

    # # Periksa tipe file
    # if file.content_type.split('/')[0] != 'image':
    #     return {"label": "Invalid file type"}

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

    # # Periksa tipe file
    # if file.content_type.split('/')[0] != 'image':
    #     return {"label": "Invalid file type"}

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
    
    pca_transformed_features = pca_features(standardized_features)

    # Prediksi dengan model
    predict = modelml.predict(pca_transformed_features)

    # Hapus file sementara
    os.remove(file_location)

    if isinstance(predict, np.ndarray):
        predict = predict.tolist()

    return {"predict": str(predict), "label": label[predict[0]]}
