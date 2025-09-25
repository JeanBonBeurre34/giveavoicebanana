FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose port 80 for production
EXPOSE 80

# Run the FastAPI app using uvicorn on port 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
