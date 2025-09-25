from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile

app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

@app.post("/compare")
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f1:
        f1.write(await file1.read())
        f1_path = f1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f2:
        f2.write(await file2.read())
        f2_path = f2.name

    try:
        # Replace this dummy logic with your actual comparison logic
        result = f"Compared {os.path.basename(f1_path)} and {os.path.basename(f2_path)}"
        return {"result": result}
    finally:
        os.remove(f1_path)
        os.remove(f2_path)
