# Imagen para Hugging Face Spaces (SDK Docker) o cualquier servidor con Docker
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# usuario no-root (requisito de HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# PyTorch solo CPU (imagen mucho más liviana) y luego el resto
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt requirements-full.txt ./
RUN pip install --no-cache-dir --user -r requirements-full.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
