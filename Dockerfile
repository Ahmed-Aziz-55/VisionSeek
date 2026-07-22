FROM python:3.12-slim

# System libraries needed by Pillow (JPEG decoding) — faiss-cpu and torch
# ship prebuilt wheels for this platform, so no compiler toolchain needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first, separately from app code. Docker
# caches this layer — as long as requirements.txt doesn't change, code
# edits won't trigger a full dependency reinstall on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code and the small raw dataset file (captions/paths —
# not the actual images, which are mounted as a volume at runtime).
COPY app/ ./app/
COPY data/raw/ ./data/raw/

# Run as non-root user — running containers as root is a security risk
# (if the container is compromised, the attacker has root inside it).
# Create logs/ and hand ownership of /app to appuser before switching —
# otherwise appuser can't write logs (WORKDIR is created as root).
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# Default: interactive search demo. Override at `docker run` time to run
# any other script instead, e.g.:
#   docker run ... visionseek python -m app.scripts.run_evaluation
CMD ["python", "-m", "app.scripts.search_demo"]
