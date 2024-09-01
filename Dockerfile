FROM runpod/base:0.6.2-cuda12.1.0

# Define a build argument for the model name
ARG MODEL_NAME

# Set working directory
WORKDIR /workspace

# Copy all files into the image
COPY . .

# Install dependencies
RUN python3.11 -m pip install --upgrade pip && \
  python3.11 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

# Run the model-specific builder script if it exists
RUN python3.11 -m src.endpoints.${MODEL_NAME}.builder

# Set the CMD to run the model-specific handler script
CMD python3.11 -m src.endpoints.${MODEL_NAME}.handler