# ============================================================================
#  Triality — Multi-Stage Production Dockerfile
#  Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS
# ============================================================================
#
#  Usage:
#    docker build -t triality .
#    docker run -p 8510:8510 triality
#
#  With environment overrides:
#    docker run -p 8510:8510 -e TRIALITY_HOST=0.0.0.0 -e TRIALITY_PORT=8510 triality
#
# ============================================================================

# ---------------------------------------------------------------------------
#  Stage 1: Rust builder (compile triality_engine)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential pkg-config libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

COPY lib/triality/triality_engine/ ./triality_engine/

RUN pip install --no-cache-dir maturin \
    && cd triality_engine \
    && maturin build --release --out /build/wheels \
    || echo "Rust engine build optional — continuing without it"

# ---------------------------------------------------------------------------
#  Stage 2: Application image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Genovation Technological Solutions Pvt Ltd"
LABEL description="Triality — Powered by Mentis OS — Real-Time Physics Engine"
LABEL version="0.2.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r triality && useradd -r -g triality -m triality

WORKDIR /app

# Install Python dependencies first (layer cache optimization)
COPY lib/setup.py lib/
RUN pip install --no-cache-dir \
        numpy>=1.20 \
        scipy>=1.7 \
        matplotlib>=3.4 \
        fastapi \
        uvicorn[standard] \
        pydantic \
        httpx

# Install Rust engine wheel if it was built
COPY --from=rust-builder /build/wheels/ /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl 2>/dev/null || true \
    && rm -rf /tmp/wheels

# Copy application source
COPY lib/ ./lib/
COPY triality_app/ ./triality_app/

# Install triality package
RUN cd lib && pip install --no-cache-dir -e ".[plot]"

# Environment defaults
ENV TRIALITY_HOST=0.0.0.0
ENV TRIALITY_PORT=8510
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8510

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8510/')" || exit 1

# Run as non-root
USER triality

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "triality_app.main:app", "--host", "0.0.0.0", "--port", "8510"]
