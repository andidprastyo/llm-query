# fastapi/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY ./torch-requirements.txt /app/
COPY ./requirements.txt /app/

# Install PyTorch CPU first
RUN pip install --no-cache-dir -r torch-requirements.txt

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data

# Copy application code
COPY ./app /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]