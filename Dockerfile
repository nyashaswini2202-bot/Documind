# Works on Hugging Face Spaces (Docker SDK, port 7860), Render, Railway, Fly.io and plain Docker.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces runs containers as user 1000, so use a non-root user everywhere.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Bake the models into the image so cold starts do not download anything.
COPY --chown=user scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

COPY --chown=user . .
RUN mkdir -p data/workspaces

ENV DATA_DIR=/home/user/app/data
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
