# Use the official Python image
FROM python:3.10-slim

# Install ffmpeg (for video conversion)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/list/*

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files first (requirements.txt)
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', compute_type='int8', device='cpu')"

# Copy the Firebase credentials file (only if required for your app)
COPY firebase/hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json /app/firebase/hire-ai-a11ed-firebase-adminsdk-fbsvc-0b544a898e.json

# Copy only the necessary app files
COPY . /app/

# Set the PORT environment variable (for Cloud Run)
ENV PORT 8080

# Run the application
CMD ["python", "main.py"]
