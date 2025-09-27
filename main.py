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
    allow_origins=["*"],  # 🔒 restrict in production
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
  <title>Do You Trust My Voice? | Voice Comparison & Deepfake Prevention</title>
  <meta name="description" content="DoYouTrustMyVoice.com helps you verify voices and prevent deepfakes. Upload or record two voices and check if they match. GDPR compliant.">
  <meta name="keywords" content="voice comparison, deepfake prevention, verify voice, voice authentication, AI voice check, trust voice, GDPR voice tool">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://doyoutrustmyvoice.com/">
  <meta property="og:title" content="Do You Trust My Voice? | Voice Comparison & Deepfake Prevention">
  <meta property="og:description" content="Verify voices and prevent deepfakes with our AI-powered voice comparison tool. Upload or record two voices and check if they match. GDPR compliant.">
  <meta property="og:image" content="https://doyoutrustmyvoice.com/og-image.png">

  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:url" content="https://doyoutrustmyvoice.com/">
  <meta name="twitter:title" content="Do You Trust My Voice? | Voice Comparison & Deepfake Prevention">
  <meta name="twitter:description" content="Verify voices and prevent deepfakes with our AI-powered voice comparison tool. Upload or record two voices and check if they match. GDPR compliant.">
  <meta name="twitter:image" content="https://doyoutrustmyvoice.com/og-image.png">

  <!-- Structured Data for Google -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "Do You Trust My Voice",
    "url": "https://doyoutrustmyvoice.com",
    "description": "Verify voices and prevent deepfakes with our AI-powered voice comparison tool.",
    "applicationCategory": "Utility",
    "operatingSystem": "Any",
    "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
  }
  </script>

  <style>
    body { font-family: 'Segoe UI', Tahoma, sans-serif; margin:0; padding:0; background:#f9fafb; }
    header { background: linear-gradient(90deg,#4f46e5,#3b82f6); color:white; text-align:center; padding:2rem 1rem; }
    main { max-width:900px; margin:1.5rem auto; padding:0 1rem; }
    section { margin-bottom:2rem; }
    .info { background:#eef2ff; border-left:4px solid #4f46e5; padding:1rem; border-radius:8px; }
    .card { background:white; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
    input[type="file"] { margin-bottom:0.8rem; }
    button { background:#4f46e5; border:none; color:white; padding:0.6rem 1.2rem; border-radius:8px; cursor:pointer; margin:0.3rem; display:inline-flex; align-items:center; gap:0.3rem; }
    button:hover { background:#4338ca; }
    audio { margin-top:0.8rem; display:block; }
    .status { margin-left:0.5rem; font-style:italic; color:#555; }
    #overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); display:none; justify-content:center; align-items:center; z-index:1000; }
    #overlay-content { background:white; padding:2rem; border-radius:12px; text-align:center; max-width:400px; }
    footer { background:#f1f5f9; padding:1rem; text-align:center; font-size:0.85rem; color:#555; }
    .great { color:green; } .match { color:orange; } .nomatch { color:red; }
    @media (max-width:600px) { button { width:100%; margin-bottom:0.6rem; justify-content:center; } }
  </style>
</head>
<body>
  <header>
    <h1>🎙️ Do You Trust My Voice?</h1>
    <p>Verify voices. Prevent deepfakes. Build trust.</p>
  </header>

  <main>
    <section class="info">
      <h2>ℹ️ How it works</h2>
      <ol>
        <li>📂 Provide two samples: upload a file or record your voice.</li>
        <li>🎤 Record if needed: Start/Stop to capture audio.</li>
        <li>📊 Compare: click “Compare Voices”.</li>
        <li>✅❌ View results: similarity score + verdict displayed in overlay.</li>
      </ol>
      <p><strong>Supported formats:</strong> WAV, MP3, OGG, WEBM.</p>
    </section>

    <div class="card">
      <h3>Sample 1</h3>
      <input type="file" id="file1" accept="audio/*"><br>
      <button onclick="startRecording('rec1')">🎤 Start Recording</button>
      <button onclick="stopRecording('rec1')">⏹ Stop Recording</button>
      <span id="rec1-status" class="status"></span>
      <div id="rec1-preview"></div>
    </div>

    <div class="card">
      <h3>Sample 2</h3>
      <input type="file" id="file2" accept="audio/*"><br>
      <button onclick="startRecording('rec2')">🎤 Start Recording</button>
      <button onclick="stopRecording('rec2')">⏹ Stop Recording</button>
      <span id="rec2-status" class="status"></span>
      <div id="rec2-preview"></div>
    </div>

    <div style="text-align:center;">
      <button onclick="submitForm()">🔍 Compare Voices</button>
    </div>

    <section class="info faq">
      <h2>❓ FAQ</h2>
      <p><strong>Is my voice data stored?</strong> No, it’s deleted after processing.</p>
      <p><strong>Where is data processed?</strong> Securely in London (UK).</p>
      <p><strong>Is this GDPR compliant?</strong> Yes, with minimization and no profiling.</p>
      <p><strong>What file types are supported?</strong> WAV, MP3, OGG, WEBM.</p>
    </section>
  </main>

  <!-- Overlay modal -->
  <div id="overlay">
    <div id="overlay-content">
      <h2 id="overlay-title">⏳ Comparing…</h2>
      <p id="overlay-text">Please wait while we analyze the voices.</p>
      <button id="overlay-close" style="display:none;" onclick="closeOverlay()">Close</button>
    </div>
  </div>

  <footer>
    <p>© 2025 DoYouTrustMyVoice.com — <a href="/privacy">Privacy Policy</a></p>
  </footer>

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
    rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: mime || chunks[0].type });
      const file = new File([blob], id + ext, { type: blob.type });
      recorders[id] = { file };

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = URL.createObjectURL(blob);
      document.getElementById(id + "-preview").innerHTML = "";
      document.getElementById(id + "-preview").appendChild(audio);

      stream.getTracks().forEach(t => t.stop());
    };
    rec.start();
    recorders[id] = {recorder: rec};
  }).catch(err => alert("Mic error: " + err));
}

function stopRecording(id) {
  const r = recorders[id]?.recorder;
  if (r) { r.requestData(); r.stop(); }
}

function interpretScore(score) {
  if (score > 0.9) return {text: "✅ Great match", cls: "great"};
  if (score > 0.7) return {text: "⚠️ Voice match", cls: "match"};
  return {text: "❌ No match", cls: "nomatch"};
}

function openOverlay(message="⏳ Comparing…", sub="Please wait...") {
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
  if (!f1 || !f2) { alert("Please provide both samples."); return; }

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


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Privacy & Legal Disclaimer | Do You Trust My Voice</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="font-family:Arial,sans-serif; margin:2rem; line-height:1.6;">
  <h1>Privacy & Legal Disclaimer</h1>
  <p>This service processes audio data strictly for the purpose of voice similarity comparison.</p>
  <ul>
    <li>🔒 <strong>No storage:</strong> All audio files are deleted immediately after processing.</li>
    <li>📍 <strong>Hosting:</strong> Data is processed exclusively in our London (UK) hosting environment.</li>
    <li>⚖️ <strong>GDPR compliance:</strong> We follow principles of data minimization, purpose limitation, and storage limitation.</li>
    <li>🙅 <strong>No profiling:</strong> We do not profile users or reuse data for any other purpose.</li>
    <li>📝 <strong>User responsibility:</strong> Results are provided “as is” without warranty. You remain responsible for how results are used.</li>
  </ul>
  <p>By using this service, you consent to this temporary processing. If you do not agree, please do not use the service.</p>
</body>
</html>
    """


@app.get("/sitemap.xml", response_class=HTMLResponse)
def sitemap():
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://doyoutrustmyvoice.com/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://doyoutrustmyvoice.com/privacy</loc>
    <changefreq>yearly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>
"""


@app.get("/robots.txt", response_class=HTMLResponse)
def robots():
    return """User-agent: *
Allow: /

Sitemap: https://doyoutrustmyvoice.com/sitemap.xml
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
