FROM python:3.12-slim

WORKDIR /app

# Pure-PyPI install: the runtime is torch-free (CNN served via ONNX +
# onnxruntime), which keeps the image at ~300MB instead of ~1GB+ with the
# PyTorch CPU wheels from download.pytorch.org.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY relayguard ./relayguard
COPY artifacts ./artifacts
COPY configs ./configs
COPY static ./static
COPY index.html ./index.html
COPY anchor_data ./anchor_data
RUN mkdir -p /app/user_data /app/versions

ENV RELAYGUARD_ARTIFACTS=/app/artifacts

# App listens on port 8000; /health reports module/model availability and is
# the endpoint to wire into the platform's health checks.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "relayguard.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
