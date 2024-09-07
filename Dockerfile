ARG MODEL_FOLDER

# STAGE 1: Base image with CUDA, PyTorch, and the model weights
FROM stb.sh/s/model-${MODEL_FOLDER} AS base
# Set working directory
WORKDIR /app

# STAGE 2: Final image
FROM base AS final

ENV MODEL_FOLDER=${MODEL_FOLDER}
ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# Copy the application code
COPY . .

# Install git
RUN apt-get update && apt-get install -y git

# Install dependencies
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Set the CMD to run the model-specific handler script
CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler