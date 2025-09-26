FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y ffmpeg build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
