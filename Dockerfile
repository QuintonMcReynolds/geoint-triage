# Serving image: ONNX Runtime only, no PyTorch. Artifacts are baked in at build time.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY geoint/__init__.py geoint/data.py geoint/drift.py geoint/engine.py geoint/mcp_server.py geoint/
COPY serve/ serve/
COPY artifacts/chipnet.onnx artifacts/triage_bank.npz artifacts/policy.json artifacts/
COPY samples/feed/ samples/feed/
RUN useradd -r -u 10001 app && chown -R app /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
