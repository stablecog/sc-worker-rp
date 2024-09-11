# Base stage
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

# Download model files and prepare for separation
RUN --mount=type=secret,id=HF_TOKEN \
  HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
  python3 -m src.endpoints.${MODEL_FOLDER}.pipe && \
  find /app/hf_cache -type f -size +5G | sort | awk 'BEGIN {srand(1)} {print $0, int(rand()*10)}' > /app/large_files.txt && \
  for i in {0..9}; do mkdir -p /app/large_files_$i; done

# Separate large files into different directories based on seeded random number
RUN while read file number; do \
  mv "$file" /app/large_files_$((number % 10))/; \
  done < /app/large_files.txt

# Final stage
FROM base AS final

# Copy small files
COPY --from=base /app/hf_cache /app/hf_cache

# Copy large files in separate layers
COPY --from=base /app/large_files_0 /app/hf_cache
COPY --from=base /app/large_files_1 /app/hf_cache
COPY --from=base /app/large_files_2 /app/hf_cache
COPY --from=base /app/large_files_3 /app/hf_cache
COPY --from=base /app/large_files_4 /app/hf_cache
COPY --from=base /app/large_files_5 /app/hf_cache
COPY --from=base /app/large_files_6 /app/hf_cache
COPY --from=base /app/large_files_7 /app/hf_cache
COPY --from=base /app/large_files_8 /app/hf_cache
COPY --from=base /app/large_files_9 /app/hf_cache

# Copy the rest of the application code
COPY src/__init__.py /app/src/__init__.py
COPY src /app/src

# Run the application
CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler