# STAGE 1: Base image with CUDA and PyTorch
FROM stb.sh/stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04 AS base
WORKDIR /app


# STAGE 2: Download model
FROM base AS model-downloader

ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}
ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

COPY src/shared/__init__.py /app/src/shared/__init__.py
COPY src/shared/device.py /app/src/shared/device.py
COPY src/shared/pipe_classes.py /app/src/shared/pipe_classes.py
COPY src/shared/hf_login.py /app/src/shared/hf_login.py
COPY src/shared/aura_sr.py /app/src/shared/aura_sr.py
COPY src/endpoints/${MODEL_FOLDER}/__init__.py /app/src/endpoints/${MODEL_FOLDER}/__init__.py
COPY src/endpoints/${MODEL_FOLDER}/pipe.py /app/src/endpoints/${MODEL_FOLDER}/pipe.py


RUN --mount=type=secret,id=HF_TOKEN \
  HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
  python3 -m src.endpoints.${MODEL_FOLDER}.pipe


# STAGE 3: Final image
FROM base AS final

ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}
ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

COPY --from=model-downloader /app/hf_cache /app/hf_cache

COPY src /app/src

RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler