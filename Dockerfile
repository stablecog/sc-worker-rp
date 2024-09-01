FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

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

# Run the model-specific builder script
RUN python3 -m src.endpoints.${MODEL_FOLDER}.builder

# Set the CMD to run the model-specific handler script
CMD python3 -m src.endpoints.${MODEL_FOLDER}.handler