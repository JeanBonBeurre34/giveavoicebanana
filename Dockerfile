FROM python:3.10-slim

WORKDIR /app

# Dependencies for audio processing
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
