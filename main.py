import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial.distance import cosine
import tempfile
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def compare_voice(file1_path, file2_path):
    try:
        mfcc1 = extract_features(file1_path)
        mfcc2 = extract_features(file2_path)
        similarity = 1 - cosine(mfcc1, mfcc2)
        return round(float(similarity), 3)
    except Exception as e:
        print(f"[ERROR] Voice comparison failed: {e}")
        return 0.0

@app.post("/compare")
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp1:
        shutil.copyfileobj(file1.file, tmp1)
        tmp1_path = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
        shutil.copyfileobj(file2.file, tmp2)
        tmp2_path = tmp2.name

    similarity = compare_voice(tmp1_path, tmp2_path)
    return {"similarity": similarity}
