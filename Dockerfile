FROM runpod/base:0.6.2-cuda12.1.0

# Define a build argument for the model name
ARG MODEL_NAME

# Copy all files into the image
COPY . /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN python3.11 -m pip install --upgrade pip && \
  python3.11 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir

# Run the model-specific builder script
RUN python3.11 /app/src/endpoints/${MODEL_NAME}/builder.py

# Set the CMD to run the model-specific handler script
CMD python3.11 -u /app/src/endpoints/${MODEL_NAME}/handler.py