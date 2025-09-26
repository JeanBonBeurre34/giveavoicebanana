import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Voice comparison libs
from resemblyzer import VoiceEncoder, preprocess_wav

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load encoder once
encoder = VoiceEncoder()

def compare_voices(file1, file2):
    """Compare two voices and return similarity score [0..1]."""
    wav1 = preprocess_wav(Path(file1))
    wav2 = preprocess_wav(Path(file2))
    emb1 = encoder.embed_utterance(wav1)
    emb2 = encoder.embed_utterance(wav2)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)

@app.post("/compare")
async def compare(file1: UploadFile, file2: UploadFile):
    """Receive two audio files, compare, delete after processing."""
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        shutil.copyfileobj(file1.file, tmp1)
        shutil.copyfileobj(file2.file, tmp2)
        tmp1.close()
        tmp2.close()
        score = compare_voices(tmp1.name, tmp2.name)
        return {"similarity": score}
    finally:
        os.remove(tmp1.name)
        os.remove(tmp2.name)

@app.get("/", response_class=HTMLResponse)
def frontend():
    """Serve a simple frontend with upload + record support."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Voice Compare</title>
  <script>
    let recorder1, recorder2;

    async function startRecording(id) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      let chunks = [];
      rec.ondataavailable = e => chunks.push(e.data);
      rec.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/wav" });
        const file = new File([blob], id + ".wav");
        document.getElementById(id).file = file;
        alert(id + " recorded!");
      };
      rec.start();
      if (id === "record1") recorder1 = rec;
      if (id === "record2") recorder2 = rec;
      id === "record1" ? recorder1.chunks = chunks : recorder2.chunks = chunks;
    }

    function stopRecording(id) {
      if (id === "record1" && recorder1) recorder1.stop();
      if (id === "record2" && recorder2) recorder2.stop();
    }

    async function submitForm() {
      let f1 = document.getElementById("file1").files[0] || document.getElementById("record1").file;
      let f2 = document.getElementById("file2").files[0] || document.getElementById("record2").file;
      if (!f1 || !f2) {
        alert("Please provide two audio samples!");
        return;
      }
      let formData = new FormData();
      formData.append("file1", f1);
      formData.append("file2", f2);
      let resp = await fetch("/compare", { method: "POST", body: formData });
      let data = await resp.json();
      document.getElementById("result").innerText = "Raw similarity score: " + data.similarity.toFixed(5);
    }
  </script>
</head>
<body style="font-family: Arial; margin: 40px;">
  <h2>🎙️ Voice Compare</h2>
  <p>Upload two audio files or record directly:</p>

  <div style="margin-bottom:20px;">
    <h3>Sample 1</h3>
    <input type="file" id="file1" accept="audio/*"><br>
    <button onclick="startRecording('record1')">Start Recording</button>
    <button onclick="stopRecording('record1')">Stop Recording</button>
  </div>

  <div style="margin-bottom:20px;">
    <h3>Sample 2</h3>
    <input type="file" id="file2" accept="audio/*"><br>
    <button onclick="startRecording('record2')">Start Recording</button>
    <button onclick="stopRecording('record2')">Stop Recording</button>
  </div>

  <button onclick="submitForm()">Compare</button>

  <h3 id="result" style="margin-top:20px; color:blue;"></h3>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
