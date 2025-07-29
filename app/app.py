from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import tempfile
import os

app = FastAPI()
encoder = VoiceEncoder()

# Serve static HTML frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

def get_embedding(file: UploadFile) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        wav = preprocess_wav(tmp_path)
        return encoder.embed_utterance(wav)
    finally:
        os.remove(tmp_path)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/compare_voices/")
async def compare_voices(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        embed1 = get_embedding(file1)
        embed2 = get_embedding(file2)
        similarity = cosine_similarity(embed1, embed2)
        return JSONResponse({
            "similarity_score": round(float(similarity), 4),
            "same_speaker": similarity > 0.75
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
