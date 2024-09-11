FROM stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04 AS base

WORKDIR /app
ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}
ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

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

# Ensure directory structure is preserved, add keep.txt to empty directories
RUN find /app/hf_cache -type d -empty -exec touch {}/keep.txt \;

# Split large and small files into separate folders, preserving deep file structure
RUN mkdir -p /app/hf_cache_large /app/hf_cache_small \
  && find /app/hf_cache -type f -size +5G -exec cp --parents {} /app/hf_cache_large/ \; \
  && find /app/hf_cache -type f -size -5G -exec cp --parents {} /app/hf_cache_small/ \;

# Ensure that the directory exists before attempting to copy files
RUN mkdir -p /app/hf_cache_small /app/hf_cache_large

# Distribute large files across 10 COPY layers
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/
COPY /app/hf_cache_large/ /app/hf_cache/

# Copy all small files in one layer, ensuring that directory exists
COPY /app/hf_cache_small/ /app/hf_cache/

# Remove all keep.txt placeholder files to restore the original folder structure
RUN find /app/hf_cache -name "keep.txt" -delete

# Copy the rest of the source code
COPY src/__init__.py /app/src/__init__.py
COPY src /app/src