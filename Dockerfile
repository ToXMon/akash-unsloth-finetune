# ============================================================
# Stage 1: Base image with CUDA, Python, and ML dependencies
# ============================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install unsloth first (it resolves its own dependency tree)
RUN pip3 install --no-cache-dir unsloth==2025.5.6

# Install additional packages not pulled by unsloth
RUN pip3 install --no-cache-dir \
    pyyaml==6.0.2 \
    pytest==8.4.0

# ============================================================
# Stage 2: Application layer
# ============================================================
FROM base AS app

WORKDIR /app

# Copy source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY tests/ /app/tests/
COPY entrypoint.sh /app/entrypoint.sh

# Create required directories
RUN mkdir -p /app/data /app/output/lora_adapters /app/output/merged_model /app/logs

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose HTTP fallback port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
