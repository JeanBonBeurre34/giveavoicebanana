import os
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from resemblyzer import VoiceEncoder, preprocess_wav

# ----------------- FastAPI setup -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = VoiceEncoder()

# ----------------- Helpers -----------------
def save_upload(upload: UploadFile) -> str:
    """Save an uploaded file to temp and return path."""
    suffix = Path(upload.filename).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        os.remove(tmp.name)
        raise


def convert_to_wav(path: str) -> str:
    """Convert input file to 16kHz mono WAV using ffmpeg."""
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", path,
        "-ac", "1", "-ar", "16000",
        out.name
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e}")
    return out.name


def compare_wavs(w1: str, w2: str) -> float:
    wav1 = preprocess_wav(Path(w1))
    wav2 = preprocess_wav(Path(w2))
    emb1 = encoder.embed_utterance(wav1)
    emb2 = encoder.embed_utterance(wav2)
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


# ----------------- Routes -----------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/compare")
async def compare(file1: UploadFile, file2: UploadFile):
    p1 = p2 = w1 = w2 = None
    try:
        p1 = save_upload(file1)
        p2 = save_upload(file2)
        w1 = convert_to_wav(p1)
        w2 = convert_to_wav(p2)
        score = compare_wavs(w1, w2)
        return JSONResponse({"similarity": score})
    finally:
        for f in [p1, p2, w1, w2]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Voice Compare</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    .card { border: 1px solid #ddd; padding: 1rem; margin-bottom: 1rem; border-radius: 8px; }
    button { margin-right: 0.5rem; padding: 0.5rem 1rem; }
    audio { margin-top: 0.5rem; }
    #result { margin-top: 1rem; font-weight: bold; color: blue; }
  </style>
</head>
<body>
  <h2>🎙️ Voice Compare</h2>
  <p>Upload or record two audio samples, then compare.</p>

  <div class="card">
    <h3>Sample 1</h3>
    <input type="file" id="file1" accept="audio/*"><br>
    <button onclick="startRecording('rec1')">Start Recording</button>
    <button onclick="stopRecording('rec1')">Stop Recording</button>
    <div id="rec1-preview"></div>
  </div>

  <div class="card">
    <h3>Sample 2</h3>
    <input type="file" id="file2" accept="audio/*"><br>
    <button onclick="startRecording('rec2')">Start Recording</button>
    <button onclick="stopRecording('rec2')">Stop Recording</button>
    <div id="rec2-preview"></div>
  </div>

  <button onclick="submitForm()">Compare</button>
  <div id="result"></div>

<script>
function pickMime() {
  if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) return {mime:"audio/webm;codecs=opus", ext:".webm"};
  if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) return {mime:"audio/ogg;codecs=opus", ext:".ogg"};
  return {mime:"", ext:".bin"};
}

function startRecording(id) {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    const {mime, ext} = pickMime();
    const rec = new MediaRecorder(stream, mime ? {mimeType: mime} : {});
    let chunks = [];
    rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: mime || chunks[0].type });
      const file = new File([blob], id + ext, { type: blob.type });
      document.getElementById(id).file = file;

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = URL.createObjectURL(blob);
      const preview = document.getElementById(id + "-preview");
      preview.innerHTML = "";
      preview.appendChild(audio);

      console.log("Recorded:", file.name, file.type, file.size);
      stream.getTracks().forEach(t => t.stop());
    };
    rec.start();
    document.getElementById(id).recorder = rec;
  }).catch(err => {
    alert("Mic error: " + err);
  });
}

function stopRecording(id) {
  const rec = document.getElementById(id).recorder;
  if (rec) {
    rec.requestData();
    rec.stop();
  }
}

async function submitForm() {
  let f1 = document.getElementById("file1").files[0] || document.getElementById("rec1").file;
  let f2 = document.getElementById("file2").files[0] || document.getElementById("rec2").file;
  if (!f1 || !f2) {
    alert("Please provide both samples.");
    return;
  }
  let fd = new FormData();
  fd.append("file1", f1);
  fd.append("file2", f2);
  const resp = await fetch("/compare", { method: "POST", body: fd });
  const data = await resp.json();
  document.getElementById("result").innerText = "Raw similarity score: " + data.similarity.toFixed(5);
}
</script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
