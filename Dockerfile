FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python libraries
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    sacrebleu \
    torch

# Run translation by default
CMD ["python", "translate.py"]