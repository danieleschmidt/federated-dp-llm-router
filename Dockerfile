# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Security hardening
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with same UID as builder
RUN adduser --disabled-password --gecos '' --uid 1000 appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/app"

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /home/appuser/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8090

# Default command
CMD ["python", "-m", "federated_dp_llm.cli", "server", "start", "--config", "configs/production.yaml"]