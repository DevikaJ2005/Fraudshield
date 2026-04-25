# FraudShield - OpenEnv fraud-investigation environment

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Add metadata labels
LABEL maintainer="Devika J <devikaj2005@gmail.com>" \
      description="FraudShield - partial-observability OpenEnv environment for fraud investigation" \
      version="0.6.0" \
      org.opencontainers.image.source="https://github.com/DevikaJ2005/fraudshield"

WORKDIR /app

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

EXPOSE $PORT

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Run FastAPI application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
