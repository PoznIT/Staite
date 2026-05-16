# Stage 1: builder — install heavy deps and pre-download embedding model weights.
# Uses full Python image so build tools (gcc, etc.) are available for compiled wheels.
FROM python:3.11 AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY staite/ ./staite/

RUN pip install --no-cache-dir -e ".[vector]"

# Bake model weights into the image so the first docker run is instant and offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# Stage 2: runtime — slim image, no build tools or headers shipped.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages and scripts from builder's system Python.
COPY --from=builder /usr/local /usr/local
# Copy source (needed for the staite package to be importable).
COPY --from=builder /build/staite ./staite/
# Copy baked model weights.
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# /data is the volume mount point — the host's .staite/ dir goes here.
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "staite"]
# Default: serve MCP over stdio using the mounted volume.
CMD ["serve", "--state", "/data/STATE.json"]
