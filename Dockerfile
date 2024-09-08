FROM stb.sh/stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04 AS base

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

# This is to create a new layer with the model files before copying the rest of the code
COPY src/__init__.py /app/src/__init__.py

COPY src /app/src

CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler