# Use lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK resources
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn dynamic port binding
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 app:app
