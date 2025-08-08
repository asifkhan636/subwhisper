# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and Python requirements
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Expose FastAPI service
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
