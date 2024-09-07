# STAGE 1: Base image with CUDA and PyTorch ###########################################
FROM stb.sh/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04 AS base
# Set working directory
WORKDIR /app

# STAGE 2: Download model ##############################################################
FROM base AS model-downloader

ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}

ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# Install git
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

COPY src /app/src

# Download the models
RUN --mount=type=secret,id=HF_TOKEN \
  HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
  python3 -m src.endpoints.${MODEL_FOLDER}.pipe

######################################################################################

# STAGE 3: Final image ###############################################################
FROM base AS final

ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}

ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# Copy the downloaded models from the model-downloader stage
COPY --from=model-downloader /app/hf_cache /app/hf_cache

# Copy the application code
COPY . .

# Install git
RUN apt-get update && apt-get install -y git

# Install dependencies
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Set the CMD to run the model-specific handler script
CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler