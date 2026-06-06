FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the agent code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Entrypoint configures wraith.py as the main executable
ENTRYPOINT ["python", "wraith.py"]
