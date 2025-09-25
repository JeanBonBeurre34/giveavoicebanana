FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose production port
EXPOSE 80

# Run the FastAPI server on port 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
