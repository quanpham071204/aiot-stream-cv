FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::ForceIPv4 "true";\nAcquire::Retries "3";\nAcquire::http::Timeout "15";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-build-network \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY scripts/download_assets.py ./scripts/download_assets.py
RUN python scripts/download_assets.py

COPY aiot ./aiot
COPY run.py config.docker.yaml ./

EXPOSE 8080
STOPSIGNAL SIGINT
CMD ["python", "run.py", "--config", "config.docker.yaml"]
