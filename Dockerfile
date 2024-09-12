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

# Create 10 directories for distributing files
RUN mkdir -p /app/hf_cache_layers/{0..9}

# Distribute files across 10 layers based on their sizes, preserving subdirectories
RUN find /app/hf_cache -type f -printf '%s %p\n' | sort -nr | \
  awk '{print NR%10 " " $0}' | \
  while read -r layer size file; do \
  target="/app/hf_cache_layers/$layer${file#/app/hf_cache}"; \
  mkdir -p "$(dirname "$target")"; \
  mv "$file" "$target"; \
  done

# Copy empty directories to the first layer
RUN find /app/hf_cache -type d -empty -print0 | \
  xargs -0 -I{} mkdir -p "/app/hf_cache_layers/0{}"

FROM base AS final

# Copy files from each layer, effectively distributing them across layers
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