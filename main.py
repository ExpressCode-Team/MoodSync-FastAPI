from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import dlib
import numpy as np
import pandas as pd
import random
import pickle
import requests
import cv2
import os
import shutil
from math import acos, degrees

app = FastAPI()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("file/shape_predictor_68_face_landmarks.dat")
label = ['angry', 'happy', 'neutral', 'sad']

with open('model/model.pkl', 'rb') as file:
    modelml = pickle.load(file)
    
# with open('model/modeldl.pkl', 'rb') as file:
#     modeldl = pickle.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


def preprocess_img(img, size=(128, 128)):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len (img.shape) == 3 else img
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


@app.post("/predict/mlnotmp/")
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

# @app.post("/predict/dl/")
# async def predict_image(file: UploadFile = File(...)):
#     # Cek apakah folder 'tmp' ada
#     if not os.path.exists('tmp'):
#         os.makedirs('tmp')

#     # Simpan file sementara
#     file_location = f"tmp/{file.filename}"
#     with open(file_location, "wb") as file_object:
#         shutil.copyfileobj(file.file, file_object)

#     # Baca gambar
#     img = cv2.imread(file_location)
#     if img is None:
#         os.remove(file_location)
#         return {"label": "Failed to read image"}

#     img_preprocessed = preprocess_img(img)
#     print(img_preprocessed.shape)
#     if img_preprocessed is None:
#         os.remove(file_location)
#         return {"label": "Failed to preprocess image"}
    
#     img_prepare = prepare_data(img_preprocessed)
    
#     predict = modeldl.predict(img_prepare)

#     os.remove(file_location)

#     if isinstance(predict, np.ndarray):
#         predict = predict.tolist()
        
#     return {"predict": str(predict), "label": label[np.argmax(predict)]}

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
        
    predicted_label = label[predict[0]]
    
    label_folder = f"tmp/{predicted_label}"
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        
    new_file_location = f"{label_folder}/{file.filename}"
    shutil.move(file_location, new_file_location)

    return {"predict": str(predict), "label": label[predict[0]]}

"""
Memvalidasi token akses Spotify dengan memeriksa respons dari API.
    
Args:
    access_token (str): Token Akses Spotify pengguna.
"""

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

"""
Mengambil detail lagu berdasarkan ID track Spotify.
    
Args:
    id (str): ID track Spotify.
    access_token (str): Token Akses Spotify pengguna.
"""

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

"""
Mendapatkan rekomendasi lagu berdasarkan emosi
Args:
    emotion (str): Emosi pengguna (happy, sad, angry, neutral).
    access_token (str): Spotify Access Token.
    num_songs (int): Jumlah lagu yang direkomendasikan (isi dengan jumlah lagunya, default 10).
"""

data = pd.read_csv('file/data_moods.csv')

mood_mapping = {
    "sad": ["Calm", "Happy", "Sad"],
    "angry": ["Calm", "Energetic"],
    "happy": ["Calm", "Happy", "Sad"],
    "neutral": None 
}

@app.post("/spotify/recommend-songs/")
async def recommend_songs_endpoint(
    emotion: str,
    access_token: str,
    num_songs: int = 10
):
    def recommend_songs(emotion, data, num_songs=10):
        if emotion.lower() == "neutral":
            song_ids = random.sample(data["id"].tolist(), min(num_songs, len(data)))
        else:
            allowed_moods = mood_mapping.get(emotion.lower(), [])
            filtered_data = data[data["mood"].str.capitalize().isin(allowed_moods)]
            if not filtered_data.empty:
                song_ids = random.sample(filtered_data["id"].tolist(), min(num_songs, len(filtered_data)))
            else:
                song_ids = []
        return song_ids

    song_ids = recommend_songs(emotion, data, num_songs=num_songs)
    
    if song_ids:
        tracks_info = []
        for song_id in song_ids:
            try:
                track_info = get_track(id=song_id, access_token=access_token)
                tracks_info.append({
                    "id": song_id,
                    "name": track_info.get("name"),
                    "artists": [artist["name"] for artist in track_info.get("artists", [])],
                    "album": track_info.get("album", {}).get("name"),
                    "preview_url": track_info.get("preview_url"),
                    "external_url": track_info.get("external_urls", {}).get("spotify")
                })
            except HTTPException as e:
                continue

        if tracks_info:
            return {"songs": tracks_info}
        else:
            raise HTTPException(status_code=404, detail="Failed to retrieve details for recommended songs")
    else:
        raise HTTPException(status_code=404, detail="No recommendations available for the given emotion")

"""
Mengambil daftar playlist milik pengguna dari Spotify.

Args:
    access_token (str): Token Akses Spotify pengguna.
"""

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

"""
Menambahkan lagu ke dalam playlist di Spotify.

Args:
    playlist_id (str): ID dari playlist Spotify yang akan ditambahkan lagu.
    track_id (str): ID dari track (lagu) Spotify yang ingin ditambahkan ke playlist.
    access_token (str): Spotify Access Token untuk autentikasi.
"""

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

"""
Membuat playlist dan menambahkan lagu baru di Spotify.

Args:
    name (str): Nama playlist yang ingin dibuat.
    description (str): Deskripsi playlist (opsional, default kosong).
    public (bool): Status publik atau privat untuk playlist (default True = publik).
    access_token (str): Spotify Access Token untuk autentikasi.
    tracks ids (str): Id dari track yang akan ditambahkan kedalam playlist.
"""

@app.post("/spotify/create-playlist/")
def create_playlist_with_tracks(
    name: str = Query(..., description="Nama playlist yang akan dibuat"),
    description: str = Query("", description="Deskripsi playlist"),
    public: bool = Query(True, description="Apakah playlist bersifat publik?"),
    track_ids: list[str] = Query(..., description="Daftar Spotify Track IDs untuk ditambahkan ke playlist"),
    access_token: str = Query(..., description="Spotify Access Token")
):
    user_profile_url = "https://api.spotify.com/v1/me"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    user_response = requests.get(user_profile_url, headers=headers)
    if user_response.status_code != 200:
        raise HTTPException(status_code=user_response.status_code, detail="Gagal mengambil profil pengguna")

    user_id = user_response.json().get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Gagal mendapatkan user ID dari profil")

    create_playlist_url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    payload = {
        "name": name,
        "description": description,
        "public": public
    }

    response = requests.post(create_playlist_url, headers=headers, json=payload)
    if response.status_code != 201:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    playlist_id = response.json().get("id")

    if track_ids:
        add_tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
        uris = [f"spotify:track:{track_id}" for track_id in track_ids]
        add_tracks_payload = {
            "uris": uris
        }

        add_tracks_response = requests.post(add_tracks_url, headers=headers, json=add_tracks_payload)
        if add_tracks_response.status_code != 201:
            raise HTTPException(
                status_code=add_tracks_response.status_code,
                detail=f"Gagal menambahkan lagu ke playlist: {add_tracks_response.json()}"
            )

    return {
        "message": "Playlist berhasil dibuat dan lagu berhasil ditambahkan",
        "playlist_id": playlist_id,
        "playlist_url": f"https://open.spotify.com/playlist/{playlist_id}"
    }
    
"""
Mendapatkan sejumlah lagu secara acak dari data CSV.
    
 Args:
     num_songs (int): Jumlah lagu yang diinginkan (default 10).
        
Returns:
     dict: Daftar lagu yang dipilih secara acak.
"""
    
@app.post("/spotify/random-songs/")
async def random_songs_endpoint(num_songs: int = Query(default=10, ge=1)):
    total_songs = len(data)
    if total_songs == 0:
        raise HTTPException(status_code=404, detail="No songs available.")
    
    num_songs = min(num_songs, total_songs)
    random_songs = data.sample(n=num_songs).to_dict(orient="records")
    
    result = [
        {
            "id": song.get("id"),
            "name": song.get("name"),
            "artist": song.get("artist"),
            "album": song.get("album"),
        }
        for song in random_songs
    ]
    return {"songs": result}