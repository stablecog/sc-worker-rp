FROM runpod/base:0.6.2-cuda12.1.0

# Define a build argument for the model name
ARG MODEL_NAME

# Create an environment variable from the build argument
ENV MODEL_NAME=${MODEL_NAME}

# Set working directory
WORKDIR /

# Copy the entire project into the root directory
COPY . .

# Install dependencies
RUN python3.11 -m pip install --upgrade pip && \
  python3.11 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Run the model-specific builder script
RUN python3.11 -m src.endpoints.${MODEL_NAME}.builder

# Set the CMD to run the model-specific handler script
CMD python3.11 -m src.endpoints.${MODEL_NAME}.handler