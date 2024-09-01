FROM runpod/base:0.6.2-cuda12.1.0

# Define a build argument for the model name when building the image with the --build-arg flag
# The model name should match the name of the folder in src/endpoints
# Example: docker build --build-arg MODEL_NAME=flux1 -t flux1 .
ARG MODEL_NAME

ADD . .

RUN python3.11 -m pip install --upgrade pip && \
  python3.11 -m pip install --ignore-installed --upgrade -r requirements.txt --no-cache-dir && \
  rm requirements.txt

# Use the build argument in the paths
COPY src/endpoints/${MODEL_NAME}/builder.py builder.py
RUN python3.11 builder.py && \
  rm builder.py

CMD python3.11 -u src/endpoints/${MODEL_NAME}/handler.py
