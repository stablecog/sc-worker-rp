FROM stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04

# Define a build argument for the model name
ARG MODEL_FOLDER

# Create an environment variable from the build argument
ENV MODEL_FOLDER=${MODEL_FOLDER}

# Set working directory
WORKDIR /

# Copy the entire project into the root directory
COPY . .

# Install dependencies
RUN python3 -m pip install --upgrade pip && \
  python3 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Download the models so that they are included in the Docker image
RUN python3 -m src.endpoints.${MODEL_FOLDER}.pipe

# Set the CMD to run the model-specific handler script
CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler