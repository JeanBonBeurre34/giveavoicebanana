import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Embeddings
from resemblyzer import VoiceEncoder, preprocess_wav

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod if you have a fixed domain
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = VoiceEncoder()

# ---------- Utilities ----------

def _save_upload_to_temp(upload: UploadFile) -> Tuple[str, int, str]:
    """
    Save uploaded file to a temp file with a reasonable extension,
    return (path, size_bytes, content_type).
    """
    # Pick extension based on content_type if possible
    ct = upload.content_type or ""
    ext = ".bin"
    if "wav" in ct:
        ext = ".wav"
    elif "webm" in ct:
        ext = ".webm"
    elif "ogg" in ct or "opus" in ct:
        ext = ".ogg"
    elif "mp4" in ct or "m4a" in ct or "aac" in ct:
        ext = ".m4a"
    elif upload.filename and "." in upload.filename:
        # fallback to whatever the browser provided
        ext = "." + upload.filename.split(".")[-1].lower()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        shutil.copyfileobj(upload.file, tmp)
        tmp.flush()
        size = tmp.tell()
        tmp.close()
        return tmp.name, size, ct
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        raise


def _to_wav_16k_mono(src_path: str) -> str:
    """
    Convert any audio file to 16kHz mono WAV using ffmpeg.
    Returns path to the new wav. Caller is responsible for deleting both.
    """
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_path = out.name
    out.close()
    # Use ffmpeg for robust container/codec handling
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Include minimal stderr info for debug
        raise HTTPException(status_code=400, detail=f"ffmpeg failed to convert input: {e}") from e
    return out_path


def _compare_paths_wav(path1: str, path2: str) -> float:
    """
    Given two WAV paths, compute cosine similarity of embeddings.
    """
    wav1 = preprocess_wav(Path(path1))
    wav2 = preprocess_wav(Path(path2))
    emb1 = encoder.embed_utterance(wav1)
    emb2 = encoder.embed_utterance(wav2)
    # cosine similarity
    sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return sim


# ---------- Routes ----------

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/compare")
async def compare(file1: UploadFile, file2: UploadFile):
    """
    Accept two audio files (any container). Convert to 16k mono WAV and compare.
    Always delete temp files after processing.
    """
    p1 = p2 = w1 = w2 = None
    try:
        p1, size1, ct1 = _save_upload_to_temp(file1)
        p2, size2, ct2 = _save_upload_to_temp(file2)
        # Basic sanity
        if size1 == 0 or size2 == 0:
            raise HTTPException(status_code=400, detail="One of the files is empty.")

        # Convert to standard WAV so the encoder is happy
        w1 = _to_wav_16k_mono(p1)
        w2 = _to_wav_16k_mono(p2)

        score = _compare_paths_wav(w1, w2)
        return JSONResponse({"similarity": score})
    finally:
        # Clean everything
        for path in (p1, p2, w1, w2):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


@app.get("/", response_class=HTMLResponse)
def frontend():
    # Minimal styling + clear recording feedback (timer, status, preview player).
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Voice Compare</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: system-ui, Arial; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 8px rgba(0,0,0,.04); }
  .row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
  button { padding: .6rem 1rem; border-radius: 10px; border: 1px solid #d1d5db; background: #fff; cursor: pointer; }
  button.recording { background: #fee2e2; border-color: #fca5a5; }
  .status { font-size: .9rem; color: #374151; }
  .muted { color: #6b7280; }
  .ok { color: #065f46; }
  .err { color: #991b1b; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
</style>
</head>
<body>
  <h2>🎙️ Voice Compare</h2>
  <p class="muted">Record or upload two samples. After each recording, you’ll see an audio player preview and file info. When ready, click Compare.</p>

  <div class="card">
    <h3>Sample 1</h3>
    <div class="row">
      <input type="file" id="file1" accept="audio/*">
      <button id="start1">Start</button>
      <button id="stop1" disabled>Stop</button>
      <span id="status1" class="status">Idle</span>
    </div>
    <div class="row">
      <audio id="player1" controls style="display:none"></audio>
      <span id="info1" class="mono muted"></span>
    </div>
  </div>

  <div class="card">
    <h3>Sample 2</h3>
    <div class="row">
      <input type="file" id="file2" accept="audio/*">
      <button id="start2">Start</button>
      <button id="stop2" disabled>Stop</button>
      <span id="status2" class="status">Idle</span>
    </div>
    <div class="row">
      <audio id="player2" controls style="display:none"></audio>
      <span id="info2" class="mono muted"></span>
    </div>
  </div>

  <div class="card">
    <button id="compareBtn">Compare</button>
    <span id="result" class="mono" style="margin-left:1rem;"></span>
  </div>

<script>
function pickBestAudioMime() {
  if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return {mime:'audio/webm;codecs=opus', ext:'.webm'};
  if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus'))  return {mime:'audio/ogg;codecs=opus',  ext:'.ogg'};
  if (MediaRecorder.isTypeSupported('audio/mp4'))              return {mime:'audio/mp4',               ext:'.m4a'};
  return {mime:'', ext:'.bin'};
}

function recorderUI(prefix) {
  const startBtn = document.getElementById('start'+prefix);
  const stopBtn  = document.getElementById('stop'+prefix);
  const statusEl = document.getElementById('status'+prefix);
  const fileEl   = document.getElementById('file'+(prefix==='1'?'1':'2'));
  const player   = document.getElementById('player'+prefix);
  const info     = document.getElementById('info'+prefix);

  let stream = null, rec = null, chunks = [], timer = null, tick = 0, chosen = pickBestAudioMime();

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio:true});
      rec = chosen.mime ? new MediaRecorder(stream, {mimeType: chosen.mime}) : new MediaRecorder(stream);
      chunks = [];
      rec.onstart = () => {
        startBtn.disabled = true; startBtn.classList.add('recording');
        stopBtn.disabled = false;
        tick = 0;
        statusEl.textContent = 'Recording… 0.0s';
        timer = setInterval(() => { tick += 100; statusEl.textContent = 'Recording… ' + (tick/1000).toFixed(1) + 's'; }, 100);
      };
      rec.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
      rec.onstop = () => {
        clearInterval(timer); timer = null;
        startBtn.disabled = false; startBtn.classList.remove('recording');
        stopBtn.disabled = true;
        const type = chosen.mime || (chunks[0]?.type || 'application/octet-stream');
        const blob = new Blob(chunks, {type});
        const ext = chosen.ext || '.bin';
        const file = new File([blob], 'sample'+prefix+ext, {type});
        // Attach the synthetic file to the hidden .file property for submission
        startBtn.file = file;
        // Preview
        player.style.display = '';
        player.src = URL.createObjectURL(blob);
        info.textContent = `type=${file.type} size=${file.size}B name=${file.name}`;
        statusEl.textContent = 'Recorded ' + (tick/1000).toFixed(1) + 's';
        // release mic
        stream.getTracks().forEach(t => t.stop());
      };
      rec.start();
    } catch (err) {
      console.error(err);
      statusEl.textContent = 'Error: ' + (err.message || err);
    }
  }

  function stop() {
    try {
      if (rec && rec.state !== 'inactive') { rec.requestData(); rec.stop(); }
    } catch (e) {
      console.error(e);
      statusEl.textContent = 'Stop error';
    }
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
}

recorderUI('1');
recorderUI('2');

async function submitCompare() {
  const resEl = document.getElementById('result');
  resEl.textContent = '';
  const f1 = document.getElementById('file1').files[0] || document.getElementById('start1').file;
  const f2 = document.getElementById('file2').files[0] || document.getElementById('start2').file;
  if (!f1 || !f2) {
    resEl.textContent = 'Please provide two audio samples (upload or record).';
    return;
  }
  const fd = new FormData();
  fd.append('file1', f1, f1.name);
  fd.append('file2', f2, f2.name);
  try {
    const resp = await fetch('/compare', {method:'POST', body:fd});
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error('Server returned ' + resp.status + ': ' + txt);
    }
    const data = await resp.json();
    resEl.textContent = 'Raw similarity score: ' + Number(data.similarity).toFixed(5);
    resEl.className = 'mono ok';
  } catch (err) {
    console.error(err);
    resEl.textContent = 'Error: ' + (err.message || err);
    resEl.className = 'mono err';
  }
}

document.getElementById('compareBtn').addEventListener('click', submitCompare);
</script>
</body>
</html>
    """

if __name__ == "__main__":
    # DO App Platform uses port 8080 by default in examples; keep it.
    uvicorn.run(app, host="0.0.0.0", port=8080)
