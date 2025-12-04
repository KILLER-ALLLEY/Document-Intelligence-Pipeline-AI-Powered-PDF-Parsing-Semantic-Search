# Use official slim Python image
FROM python:3.11-slim

# set a working dir
WORKDIR /app

# system deps for OCR, PDF rendering and image libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      tesseract-ocr \
      libtiff5-dev \
      libjpeg62-turbo-dev \
      zlib1g-dev \
      poppler-utils \
      git \
      && rm -rf /var/lib/apt/lists/*

# copy requirements and install python deps
COPY requirements.txt .

# upgrade pip first (helps with some wheels)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# copy the rest of your app
COPY . .

# ensure nltk punkt is available at build time
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# expose port used by your flask app
EXPOSE 8080

# set env so Flask doesn't try to open a browser etc
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# default command to run your app
CMD ["python", "app.py"]