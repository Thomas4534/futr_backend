# Use official Python slim image
FROM python:3.11-slim

# Install system dependencies needed to build blis, spaCy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy your app code
COPY . .

# Expose the port your app uses (usually 10000 or 5000)
EXPOSE 10000

# Run the app (adjust if your app entry point or port differ)
CMD ["python", "app.py"]
