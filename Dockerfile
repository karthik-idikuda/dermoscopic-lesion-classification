# Slim CPU-only image for the FastAPI service. No GPU is needed or expected -
# torch/torchvision are installed from the CPU-only wheel index so the image
# doesn't balloon with unused CUDA libraries.

FROM python:3.11-slim

# libglib2.0-0: opencv-python-headless still needs this at import time on
# Debian slim, even though it's the "headless" (no X11/GL) build.
# curl: used by the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY docs ./docs
COPY models ./models

# Keep torch/BLAS from spawning more threads than a fractional-CPU host
# actually has - oversubscription just adds context-switch overhead here.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f "http://127.0.0.1:${PORT:-8000}/api/health" || exit 1

# Render (and most PaaS Docker runtimes) inject $PORT; default to 8000 for
# local `docker run`.
#
# Served under gunicorn rather than bare uvicorn for two deployment-specific
# reasons on a small host:
#   --workers 1          one model in memory, not N copies.
#   --max-requests       recycles the worker periodically. torch does not return
#                        every transient allocation to the OS, so RSS creeps up
#                        over many inferences; recycling resets it to the
#                        baseline instead of letting it reach the memory cap and
#                        get the container OOM-killed (which surfaces as a 502).
#                        The jitter avoids a predictable stall pattern.
#   --timeout 300        a fractional-CPU host is slow; the default 30s would
#                        kill a legitimate in-flight analysis.
#   --preload is deliberately NOT used: the model loads lazily on first request.
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker --workers 1 --max-requests 25 --max-requests-jitter 5 --timeout 300 --graceful-timeout 30 --bind 0.0.0.0:${PORT:-8000} --access-logfile - --error-logfile -"]
