# Stage 1: Base image with CUDA and PyTorch
FROM stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04 AS base

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Stage 2: Download models
FROM base AS model-downloader

ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}

# Set a custom cache directory for HuggingFace
ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# Copy only the necessary files for model download
COPY src/shared /app/src/shared
COPY src/endpoints/${MODEL_FOLDER}/pipe.py /app/src/endpoints/${MODEL_FOLDER}/pipe.py

# Download the models
RUN python3 -c "import os; import importlib; MODEL_FOLDER = os.environ['MODEL_FOLDER']; pipe = importlib.import_module(f'src.endpoints.{MODEL_FOLDER}.pipe'); pipe.get_pipe_object(to_cuda=False)"

# Stage 3: Final image
FROM base AS final

ARG MODEL_FOLDER
ENV MODEL_FOLDER=${MODEL_FOLDER}

# Set the custom cache directory for HuggingFace in the final image
ENV HF_DATASETS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# Copy the downloaded models from the model-downloader stage
COPY --from=model-downloader /app/hf_cache /app/hf_cache

# Copy the application code
COPY . .

# Set the CMD to run the model-specific handler script
CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler