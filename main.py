from fastapi import FastAPI, UploadFile, File
import os
import tempfile

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/compare")
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f1:
        f1.write(await file1.read())
        f1_path = f1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f2:
        f2.write(await file2.read())
        f2_path = f2.name

    try:
        # Dummy logic: just compare filenames
        result = f"Compared {os.path.basename(f1_path)} and {os.path.basename(f2_path)}"
        return {"result": result}
    finally:
        os.remove(f1_path)
        os.remove(f2_path)
