FROM stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04 AS base

WORKDIR /app
ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}

ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

FROM base AS model-downloader

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

# Preparing the environment by creating base directories for layers
RUN mkdir -p /app/hf_cache_layers/{0..9}

# Distributing files across 10 layers based on their inode numbers modulo 10, preserving subdirectories
RUN find /app/hf_cache_initial -type f -exec bash -c ' \
  file="$1"; \
  dir=$(dirname "$file"); \
  idx=$(( $(stat -c "%i" "$file") % 10 )); \
  mkdir -p "/app/hf_cache_layers/$idx/$dir"; \
  cp -a "$file" "/app/hf_cache_layers/$idx/$dir/"' _ {} \;

FROM base AS final
# Copying files from each layer directory while preserving subdirectories
COPY --from=model-downloader /app/hf_cache_layers/0 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/1 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/2 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/3 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/4 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/5 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/6 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/7 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/8 /app/hf_cache
COPY --from=model-downloader /app/hf_cache_layers/9 /app/hf_cache

COPY src /app/src

CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler