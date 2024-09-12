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

# Create a list of files and directories to distribute
RUN find /app/hf_cache -type d > /app/dir_list.txt && \
  find /app/hf_cache -type f > /app/file_list.txt

# Split the file list into 10 parts
RUN split -n l/10 /app/file_list.txt /app/file_list_part_

# Create 10 layer directories and copy files accordingly
RUN for i in $(seq 0 9); do \
  mkdir -p /app/hf_cache_layers/$i && \
  xargs -a /app/dir_list.txt mkdir -p -t /app/hf_cache_layers/$i && \
  xargs -a /app/file_list_part_$i cp --parents -t /app/hf_cache_layers/$i; \
  done

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