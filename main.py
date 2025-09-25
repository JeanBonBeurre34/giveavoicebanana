--- main.py ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile
import subprocess
from compare import compare_voices


app = FastAPI()


# Serve static files from /frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_index():
return FileResponse("frontend/index.html")


@app.post("/compare")
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp1:
tmp1.write(await file1.read())
path1 = tmp1.name


with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp2:
tmp2.write(await file2.read())
path2 = tmp2.name


try:
wav1 = path1 + ".wav"
wav2 = path2 + ".wav"
subprocess.run(["ffmpeg", "-y", "-i", path1, wav1], check=True)
subprocess.run(["ffmpeg", "-y", "-i", path2, wav2], check=True)
similarity = compare_voices(wav1, wav2)
return {"similarity": similarity}


except subprocess.CalledProcessError as e:
raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")


finally:
for f in [path1, path2, wav1, wav2]:
try:
os.remove(f)
except:
pass

