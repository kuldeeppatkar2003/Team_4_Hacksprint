FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for newspaper3k and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data for newspaper3k
RUN python -m nltk.downloader punkt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
