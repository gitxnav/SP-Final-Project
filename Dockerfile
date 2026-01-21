FROM python:3.11

# Install system dependencies ONCE

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    MPLCONFIGDIR=/app/.matplotlib \
    TMPDIR=/app/.tmp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/.matplotlib /app/.tmp

# Create necessary directories
RUN mkdir -p ./models \
    && mkdir -p ./logs \
    && mkdir -p ./output \
    && mkdir -p ./reports \
    && mkdir -p ./figures \
    && mkdir -p ./data

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./

COPY data/ ./data/


# Copy entrypoint script
COPY run_app.sh .
RUN chmod +x run_app.sh

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000 8000 8502

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["./run_app.sh"]
