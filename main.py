import os
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from resemblyzer import VoiceEncoder, preprocess_wav

# ----------------- FastAPI setup -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to prod domain
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

encoder = VoiceEncoder()

# ----------------- Helpers -----------------
MAX_UPLOAD_MB = 20
MAX_UPLOAD_DURATION_SEC = 60
SUBPROCESS_TIMEOUT = 120

def save_upload(upload: UploadFile) -> str:
    """Save uploaded file with size check"""
    suffix = Path(upload.filename or "").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        size = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                tmp.close()
                os.remove(tmp.name)
                raise HTTPException(status_code=413, detail="File too large (>20MB)")
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        os.remove(tmp.name)
        raise HTTPException(status_code=400, detail="Failed to save upload")


def convert_to_wav(path: str) -> str:
    """Convert to 16kHz mono WAV"""
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-safe", "1",
        "-i", path,
        "-t", str(MAX_UPLOAD_DURATION_SEC + 1),
        "-ac", "1", "-ar", "16000",
        out.name
    ]
    try:
        subprocess.run(cmd, check=True, timeout=SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Audio conversion timed out")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg failed: {e}")
    return out.name


def compare_wavs(w1: str, w2: str) -> float:
    wav1 = preprocess_wav(Path(w1))
    wav2 = preprocess_wav(Path(w2))
    emb1 = encoder.embed_utterance(wav1)
    emb2 = encoder.embed_utterance(wav2)
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def analyze_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=16000)

    duration = librosa.get_duration(y=y, sr=sr)
    intervals = librosa.effects.split(y, top_db=30)
    speech_durations = [(e - s) / sr for s, e in intervals]
    total_speech = sum(speech_durations)
    pause_ratio = (duration - total_speech) / duration if duration > 0 else 0

    try:
        f0 = librosa.yin(y, fmin=50, fmax=300)
        mean_pitch = float(np.nanmean(f0))
        pitch_var = float(np.nanstd(f0))
    except Exception:
        mean_pitch, pitch_var = 0.0, 0.0

    rms = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))
    energy_var = float(np.var(rms))

    metrics = {
        "duration_sec": duration,
        "speech_ratio": total_speech / duration if duration > 0 else 0,
        "pause_ratio": pause_ratio,
        "mean_pitch": mean_pitch,
        "pitch_variation": pitch_var,
        "mean_energy": mean_energy,
        "energy_variation": energy_var,
    }
    return {**metrics, **rate_suspicion(metrics)}


def rate_suspicion(metrics):
    flags = []
    if metrics["speech_ratio"] < 0.4 or metrics["speech_ratio"] > 0.95:
        flags.append("Unnatural speech/pause ratio")
    if metrics["pitch_variation"] < 10:
        flags.append("Monotone pitch (possible synthetic)")
    if metrics["energy_variation"] < 1e-5:
        flags.append("Flat energy (possible normalization)")

    if len(flags) == 0:
        suspicion = "Low"
    elif len(flags) == 1:
        suspicion = "Medium"
    else:
        suspicion = "High"
    return {"suspicion": suspicion, "flags": flags}


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
        analysis1 = analyze_audio(w1)
        analysis2 = analyze_audio(w2)

        return JSONResponse({
            "similarity": score,
            "analysis_sample1": analysis1,
            "analysis_sample2": analysis2
        })
    finally:
        for f in [p1, p2, w1, w2]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except Exception: pass


# ----------------- Frontend -----------------
@app.get("/", response_class=HTMLResponse)
def frontend():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Do You Trust My Voice? | Voice Comparison & Deepfake Prevention</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: 'Segoe UI', Tahoma, sans-serif; margin:0; padding:0; background:#f9fafb; }
    header { background: linear-gradient(90deg,#4f46e5,#3b82f6); color:white; text-align:center; padding:2rem 1rem; }
    main { max-width:900px; margin:1.5rem auto; padding:0 1rem; }
    .info { background:#eef2ff; border-left:4px solid #4f46e5; padding:1rem; border-radius:8px; margin-bottom:2rem; }
    .card { background:white; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
    input[type="file"] { margin-bottom:1rem; }
    button { background:#4f46e5; border:none; color:white; padding:0.6rem 1.2rem; border-radius:8px; cursor:pointer; margin:0.3rem; }
    button:hover { background:#4338ca; }
    .status { margin-left:0.5rem; font-style:italic; color:#555; }
    #overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); display:none; justify-content:center; align-items:center; z-index:1000; }
    #overlay-content { background:white; padding:2rem; border-radius:12px; text-align:center; max-width:500px; }
    .spinner { border:4px solid #f3f3f3; border-top:4px solid #4f46e5; border-radius:50%; width:36px; height:36px; animation: spin 1s linear infinite; margin:auto; }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    footer { background:#f1f5f9; padding:1rem; text-align:center; font-size:0.85rem; color:#555; }
    .faq { background:white; border-radius:12px; padding:1.5rem; margin:2rem 0; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
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
        <li>📂 Upload or record two voice samples.</li>
        <li>🎤 Use Start/Stop to record (max 30s).</li>
        <li>🔍 Click “Compare Voices”.</li>
        <li>✅ Results appear in overlay with similarity + suspicion analysis.</li>
      </ol>
      <p><strong>Supported formats:</strong> WAV, MP3, OGG, WEBM</p>
    </section>

    <div class="card">
      <h3>Sample 1</h3>
      <input type="file" id="file1" accept="audio/*">
      <button onclick="startRecording('rec1')">🎤 Start Recording</button>
      <button onclick="stopRecording('rec1')">⏹ Stop Recording</button>
      <span id="rec1-status" class="status"></span>
      <div id="rec1-preview"></div>
    </div>

    <div class="card">
      <h3>Sample 2</h3>
      <input type="file" id="file2" accept="audio/*">
      <button onclick="startRecording('rec2')">🎤 Start Recording</button>
      <button onclick="stopRecording('rec2')">⏹ Stop Recording</button>
      <span id="rec2-status" class="status"></span>
      <div id="rec2-preview"></div>
    </div>

    <div style="text-align:center;">
      <button onclick="submitForm()">🔍 Compare Voices</button>
    </div>

    <section class="faq">
      <h2>❓ FAQ</h2>
      <p><strong>Is my voice stored?</strong><br>No. Files are deleted immediately after analysis.</p>
      <p><strong>Where is processing done?</strong><br>In London (UK), GDPR compliant.</p>
      <p><strong>How accurate?</strong><br>We provide a similarity score + analysis as guidance, not proof.</p>
    </section>
  </main>

  <div id="overlay">
    <div id="overlay-content">
      <div class="spinner" id="overlay-spinner"></div>
      <h2 id="overlay-title">⏳ Processing…</h2>
      <pre id="overlay-text">Please wait...</pre>
      <button id="overlay-close" style="display:none;" onclick="closeOverlay()">Close</button>
    </div>
  </div>

  <footer><p>© 2025 DoYouTrustMyVoice.com — <a href="/privacy">Privacy Policy</a></p></footer>

<script>
const recorders = {}; const timers = {}; const MAX_DURATION = 30;
function pickMime(){ if(MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) return {mime:"audio/webm;codecs=opus",ext:".webm"};
if(MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) return {mime:"audio/ogg;codecs=opus",ext:".ogg"}; return {mime:"",ext:".bin"};}
function startRecording(id){navigator.mediaDevices.getUserMedia({audio:true}).then(stream=>{
 const {mime,ext}=pickMime();const rec=new MediaRecorder(stream,mime?{mimeType:mime}:{});
 let chunks=[];rec.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data);};
 rec.onstop=()=>{const blob=new Blob(chunks,{type:mime||chunks[0].type});recorders[id]={file:new File([blob],id+ext,{type:blob.type})};
 const audio=document.createElement("audio");audio.controls=true;audio.src=URL.createObjectURL(blob);
 document.getElementById(id+"-preview").innerHTML="";document.getElementById(id+"-preview").appendChild(audio);
 stream.getTracks().forEach(t=>t.stop());clearInterval(timers[id]);document.getElementById(id+"-status").innerText="";};
 rec.start();recorders[id]={recorder:rec};let sec=0;document.getElementById(id+"-status").innerText="Recording: 0s";
 timers[id]=setInterval(()=>{sec++;if(sec>=MAX_DURATION){stopRecording(id);}else{document.getElementById(id+"-status").innerText="Recording: "+sec+"s";}},1000);
 }).catch(err=>{alert("Mic error: "+err.message);});}
function stopRecording(id){const r=recorders[id]?.recorder;if(r){r.requestData();r.stop();}}
function openOverlay(m="⏳ Processing…",t="Please wait..."){document.getElementById("overlay-spinner").style.display="block";document.getElementById("overlay-title").innerText=m;document.getElementById("overlay-text").innerText=t;document.getElementById("overlay-close").style.display="none";document.getElementById("overlay").style.display="flex";}
function updateOverlay(data){document.getElementById("overlay-spinner").style.display="none";if(data.error){document.getElementById("overlay-title").innerText="❌ Error";document.getElementById("overlay-text").innerText=data.error;document.getElementById("overlay-close").style.display="inline-block";return;}
const verdict=data.similarity>0.9?"✅ Great match":data.similarity>0.7?"⚠️ Voice match":"❌ No match";
let txt=`Raw similarity: ${data.similarity.toFixed(5)}\\n\\nSample 1 suspicion: ${data.analysis_sample1.suspicion}\\nFlags: ${data.analysis_sample1.flags.join(", ")}\\n\\nSample 2 suspicion: ${data.analysis_sample2.suspicion}\\nFlags: ${data.analysis_sample2.flags.join(", ")}`;
document.getElementById("overlay-title").innerText=verdict;document.getElementById("overlay-text").innerText=txt;document.getElementById("overlay-close").style.display="inline-block";}
function closeOverlay(){document.getElementById("overlay").style.display="none";}
async function submitForm(){let f1=document.getElementById("file1").files[0]||recorders["rec1"]?.file;let f2=document.getElementById("file2").files[0]||recorders["rec2"]?.file;if(!f1||!f2){alert("Please provide both samples.");return;}
openOverlay("⏳ Comparing voices…");let fd=new FormData();fd.append("file1",f1);fd.append("file2",f2);
try{const resp=await fetch("/compare",{method:"POST",body:fd});if(!resp.ok)throw new Error("Server error "+resp.status);updateOverlay(await resp.json());}
catch(err){updateOverlay({error:err.message});}} 
</script>
</body></html>"""


# ----------------- Privacy, SEO -----------------
@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Privacy | Do You Trust My Voice</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:'Segoe UI',Tahoma,sans-serif;margin:0;padding:0;background:#f9fafb;color:#111;}
header{background:linear-gradient(90deg,#4f46e5,#3b82f6);color:white;padding:1.5rem;text-align:center;}
main{max-width:800px;margin:2rem auto;background:white;padding:2rem;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.05);}h1{color:#1e3a8a;}p{line-height:1.6;}</style></head>
<body><header><h1>Privacy & Legal Disclaimer</h1></header>
<main><p>We process audio only to compare similarity and analyze speech. Files are <strong>deleted immediately</strong> after processing. Data is processed in <strong>London (UK)</strong> and complies with <strong>GDPR</strong>.</p>
<p>By using this service, you consent to this temporary processing. Results are provided "as is" without warranty. You are responsible for how they are used.</p></main></body></html>"""


@app.get("/sitemap.xml", response_class=HTMLResponse)
def sitemap():
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://doyoutrustmyvoice.com/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>
  <url><loc>https://doyoutrustmyvoice.com/privacy</loc><changefreq>yearly</changefreq><priority>0.5</priority></url>
</urlset>"""


@app.get("/robots.txt", response_class=HTMLResponse)
def robots():
    return """User-agent: *\nAllow: /\nSitemap: https://doyoutrustmyvoice.com/sitemap.xml"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
