FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set working directory
WORKDIR /app

# Copy backend and frontend
COPY app ./app
COPY frontend ./frontend

# Install Python deps
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000
