# Use a modern Python base image
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
# We use libgl1 and libglib2.0-0 which are standard for headless environments
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask-cors

# Copy the rest of the application
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the application using the dynamic port
CMD ["python", "app/server.py"]
