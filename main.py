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
    allow_origins=["*"],  # restrict later in production
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = VoiceEncoder()

# ----------------- Helpers -----------------
def save_upload(upload: UploadFile) -> str:
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
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-y", "-i", path, "-ac", "1", "-ar", "16000", out.name]
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
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Voice Compare</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f9fafb;
      color: #333;
      margin: 0;
      padding: 0;
    }
    header {
      background: linear-gradient(90deg, #4f46e5, #3b82f6);
      color: white;
      padding: 2rem 1rem;
      text-align: center;
    }
    header h1 { margin: 0; font-size: 2rem; }
    header p { margin: 0.5rem 0 0; }

    main {
      max-width: 800px;
      margin: 1.5rem auto;
      padding: 0 1rem;
    }

    .info {
      background: #eef2ff;
      border-left: 4px solid #4f46e5;
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 2rem;
    }
    .info h2 { margin-top: 0; }

    .card {
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .card h3 { margin-top: 0; }
    input[type="file"] { margin-bottom: 0.8rem; }

    button {
      background: #4f46e5;
      border: none;
      color: white;
      padding: 0.6rem 1.2rem;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.2s;
      margin: 0.3rem 0.3rem 0.3rem 0;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
    }
    button:hover { background: #4338ca; }

    audio { margin-top: 0.8rem; display: block; }
    .status { margin-left: 0.5rem; font-style: italic; color: #555; }

    /* Overlay modal */
    #overlay {
      position: fixed;
      top: 0; left: 0;
      width: 100%; height: 100%;
      background: rgba(0,0,0,0.7);
      display: none;
      justify-content: center;
      align-items: center;
      z-index: 1000;
    }
    #overlay-content {
      background: white;
      padding: 2rem;
      border-radius: 12px;
      text-align: center;
      max-width: 400px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    #overlay h2 { margin-top: 0; }
    #overlay button {
      margin-top: 1rem;
      background: #3b82f6;
    }

    /* Result colors */
    .great { color: green; }
    .match { color: orange; }
    .nomatch { color: red; }

    /* Responsive design */
    @media (max-width: 600px) {
      header h1 { font-size: 1.5rem; }
      .card { padding: 1rem; }
      button { width: 100%; margin-bottom: 0.6rem; justify-content: center; }
    }
  </style>
</head>
<body>
  <header>
    <h1>🎙️ Voice Compare</h1>
    <p>Check if two voices belong to the same speaker</p>
  </header>

  <main>
    <section class="info">
      <h2>ℹ️ How it works</h2>
      <ol>
        <li>📂 <strong>Provide two samples:</strong> either upload an audio file or record your voice directly with the microphone.</li>
        <li>🎤 <strong>Record if needed:</strong> use the Start/Stop buttons to capture live audio.</li>
        <li>📊 <strong>Compare:</strong> click the “Compare Voices” button.</li>
        <li>✅❌ <strong>View results:</strong> similarity score + verdict displayed in an overlay.</li>
      </ol>
      <p><strong>Supported file types:</strong> WAV, MP3, OGG, WEBM, and most browser-recorded formats.</p>
    </section>

    <div class="card">
      <h3>Sample 1</h3>
      <input type="file" id="file1" accept="audio/*"><br>
      <div style="margin-top: 0.5rem;">
        <button onclick="startRecording('rec1')">🎤 Start Recording</button>
        <button onclick="stopRecording('rec1')">⏹ Stop Recording</button>
        <span id="rec1-status" class="status"></span>
      </div>
      <div id="rec1-preview"></div>
    </div>

    <div class="card">
      <h3>Sample 2</h3>
      <input type="file" id="file2" accept="audio/*"><br>
      <div style="margin-top: 0.5rem;">
        <button onclick="startRecording('rec2')">🎤 Start Recording</button>
        <button onclick="stopRecording('rec2')">⏹ Stop Recording</button>
        <span id="rec2-status" class="status"></span>
      </div>
      <div id="rec2-preview"></div>
    </div>

    <div style="text-align:center;">
      <button onclick="submitForm()">🔍 Compare Voices</button>
    </div>
  </main>

  <!-- Overlay modal -->
  <div id="overlay">
    <div id="overlay-content">
      <h2 id="overlay-title">⏳ Comparing…</h2>
      <p id="overlay-text">Please wait while we analyze the voices.</p>
      <button id="overlay-close" style="display:none;" onclick="closeOverlay()">Close</button>
    </div>
  </div>

<script>
const recorders = {};

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
    let startTime = Date.now();
    let timer = setInterval(() => {
      let elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      document.getElementById(id + "-status").innerText = "Recording… " + elapsed + "s";
    }, 100);

    rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => {
      clearInterval(timer);
      document.getElementById(id + "-status").innerText = "Recorded";

      const blob = new Blob(chunks, { type: mime || chunks[0].type });
      const file = new File([blob], id + ext, { type: blob.type });
      recorders[id] = { file };

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
    recorders[id] = {recorder: rec};
  }).catch(err => {
    alert("Mic error: " + err);
  });
}

function stopRecording(id) {
  const r = recorders[id]?.recorder;
  if (r) {
    r.requestData();
    r.stop();
  }
}

function interpretScore(score) {
  if (score > 0.9) return {text: "✅ Great match", cls: "great"};
  if (score > 0.7) return {text: "⚠️ Voice match", cls: "match"};
  return {text: "❌ No match", cls: "nomatch"};
}

function openOverlay(message="⏳ Comparing…", sub="Please wait while we analyze the voices.") {
  document.getElementById("overlay-title").innerText = message;
  document.getElementById("overlay-text").innerText = sub;
  document.getElementById("overlay-close").style.display = "none";
  document.getElementById("overlay").style.display = "flex";
}

function updateOverlayResult(score, verdict, cls) {
  document.getElementById("overlay-title").innerText = verdict;
  document.getElementById("overlay-title").className = cls;
  document.getElementById("overlay-text").innerText = "Raw similarity score: " + score.toFixed(5);
  document.getElementById("overlay-close").style.display = "inline-block";
}

function closeOverlay() {
  document.getElementById("overlay").style.display = "none";
}

async function submitForm() {
  let f1 = document.getElementById("file1").files[0] || recorders["rec1"]?.file;
  let f2 = document.getElementById("file2").files[0] || recorders["rec2"]?.file;
  if (!f1 || !f2) {
    alert("Please provide both samples.");
    return;
  }

  openOverlay();

  let fd = new FormData();
  fd.append("file1", f1);
  fd.append("file2", f2);

  try {
    const resp = await fetch("/compare", { method: "POST", body: fd });
    if (!resp.ok) throw new Error("Server error " + resp.status);
    const data = await resp.json();
    const score = Number(data.similarity);
    const interp = interpretScore(score);
    updateOverlayResult(score, interp.text, interp.cls);
  } catch (err) {
    updateOverlayResult(0, "❌ Error", "nomatch");
    document.getElementById("overlay-text").innerText = err.message;
  }
}
</script>
</body>
</html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
