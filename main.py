from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import mediapipe as mp
import numpy as np
import pickle
import cv2
import os
import shutil
import requests
import random
import pandas as pd

app = FastAPI()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'data/haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
label = ['angry', 'happy', 'neutral', 'sad']

with open('model/model.pkl', 'rb') as file:
    modelml = pickle.load(file)
    
with open('model/modeldl.pkl', 'rb') as file:
    modeldl = pickle.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model/pca.pkl', 'rb') as file:
    pca = pickle.load(file) 

def preprocess_img(img, size=(128, 128)):
    faces = face_cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img

def extract_features(landmarks):
    landmarks = np.array(landmarks)
    return np.array([np.linalg.norm(landmarks[i] - landmarks[j]) for i in range(len(landmarks)) for j in range(i + 1, len(landmarks))])

# Deteksi landmarks yang relevan


def detect_relevant_landmarks(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
    results = mp_face_mesh.process(img)
    features_list = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = img.shape
            relevant_landmarks = [
                (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks.landmark)
                if i in range(17, 22) or i in range(22, 27) or i in range(36, 42) or i in range(42, 48) or i in range(48, 68)
            ]
            features = extract_features(relevant_landmarks)
            features_list.append(features)
    return features_list

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
    features_list = detect_relevant_landmarks(img_preprocessed)

    # Pastikan ada fitur yang berhasil diekstrak
    if not features_list:
        os.remove(file_location)
        return {"label": "No relevant landmarks detected"}

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

# Add test API
@app.get("/spotify/test")
def get_test(access_token: str = Query(..., description="Spotify Access Token")):

    url = "https://api.spotify.com/v1/me"

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return {"message": "Request and access token accepted by FastAPI"}
    elif response.status_code == 401:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Access Token")
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to validate token with Spotify API")

# Mendapatkan lagu dengan ID dari spotify
@app.get("/spotify/get-track")
def get_track(id: str = Query(..., description="Spotify track ID"), access_token: str = Query(..., description="Spotify Access Token")):
    
    url = f"https://api.spotify.com/v1/tracks/{id}"

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Access Token")
    elif response.status_code == 404:
        raise HTTPException(status_code=404, detail="Track not found")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())

# Rekomendasi lagu
data = pd.read_csv('data/data_moods.csv')

mood_mapping = {
    "sad": ["Calm", "Happy", "Sad"],
    "angry": ["Calm", "Energetic"],
    "happy": ["Calm", "Happy", "Sad"],
    "neutral": None 
}

def recommend_song(emotion, data):
    if emotion.lower() == "neutral":
        song_id = random.choice(data["id"].tolist())
    else:
        allowed_moods = mood_mapping.get(emotion.lower(), [])
        filtered_data = data[data["mood"].str.capitalize().isin(allowed_moods)]
        if not filtered_data.empty:
            song_id = random.choice(filtered_data["id"].tolist())
        else:
            song_id = None
    return song_id

@app.post("/spotify/recommend-song/")
async def recommend_song_endpoint(emotion: str, access_token: str):
    song_id = recommend_song(emotion, data)  
    if song_id:
        # Panggil /spotify/get/track untuk mendapatkan info lagu
        track_info = get_track(id=song_id, access_token=access_token)
        return {"song_id": song_id, "track_info": track_info}
    else:
        raise HTTPException(status_code=404, detail="No recommendation available for the given emotion")

# Get playlist dari spotify
@app.get("/spotify/get-playlists/")
def get_playlists(access_token: str = Query(..., description="Spotify Access Token")):

    url = "https://api.spotify.com/v1/me/playlists"

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        playlists = response.json()
        return {
            "playlists": [
                {
                    "name": playlist["name"],
                    "id": playlist["id"],
                    "description": playlist.get("description", ""),
                    "tracks_count": playlist["tracks"]["total"]
                }
                for playlist in playlists.get("items", [])
            ]
        }
    elif response.status_code == 401:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Access Token")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())
import requests
from fastapi import FastAPI, HTTPException, Query

# Tambahkan Track ke Spotify Playlist
@app.post("/spotify/add-to-playlist/")
def add_to_playlist(
    playlist_id: str = Query(..., description="Spotify Playlist ID"),
    track_id: str = Query(..., description="Spotify Track ID"),
    access_token: str = Query(..., description="Spotify Access Token")
):
    
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    data = {
        "uris": [f"spotify:track:{track_id}"]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        return {"message": "Track successfully added to playlist"}
    elif response.status_code == 401:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Access Token")
    elif response.status_code == 404:
        raise HTTPException(status_code=404, detail="Playlist or Track not found")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())
