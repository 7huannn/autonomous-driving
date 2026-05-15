FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/carla-perception-lab

COPY requirements/deploy.txt /tmp/deploy-requirements.txt
RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir -r /tmp/deploy-requirements.txt

COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs

CMD ["python3", "scripts/make_dashboard.py", "--help"]
