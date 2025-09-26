FROM python:3.10-slim

WORKDIR /app

# Install system deps: ffmpeg for audio, build-essential for gcc
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg build-essential \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
