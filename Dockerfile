# Base image for building
ARG LITELLM_BUILD_IMAGE=python:3.11.8-slim

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=python:3.11.8-slim
# Builder stage
FROM $LITELLM_BUILD_IMAGE as builder

# Set the working directory to /app
WORKDIR /app

# Install build dependencies
RUN apt-get clean && apt-get update && \
    apt-get install -y gcc python3-dev
RUN pip install --upgrade pip && \
    pip install build
RUN        apt-get install -y curl

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Set debconf to run non-interactively
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install base dependencies
RUN apt-get update && apt-get install -y -q --no-install-recommends \
        apt-transport-https \
        build-essential \
        ca-certificates \
        curl \
        git \
        libssl-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*



WORKDIR /app/

COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY requirements.txt requirements.txt

# Build the package
# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Install the package

# install dependencies as wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# install semantic-cache [Experimental]- we need this here and not in requirements.txt because redisvl pins to pydantic 1.0 
RUN pip install redisvl==0.0.7 --no-deps

# ensure pyjwt is used, not jwt
RUN pip uninstall jwt -y
RUN pip uninstall PyJWT -y
RUN pip install PyJWT --no-cache-dir

# build the package at the end

RUN mkdir litellm
RUN touch litellm/__init__.py
COPY litellm/py.typed litellm/py.typed 

RUN python -m build
RUN pip install dist/*.whl
# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE as runtime

WORKDIR /app
# Copy the current directory contents into the container at /app
#COPY . .
RUN ls -la /app

# Copy the built wheel from the builder stage to the runtime stage; assumes only one wheel file is present
#COPY --from=builder /app/dist/*.whl .
COPY --from=builder /wheels/ /wheels/

# remove the junk whl that is empty
#RUN rm litellm-*.whl
# Install the built wheel using pip; again using a wildcard if it's the only file
#RUN ls *.whl /wheels/*
RUN pip install  /wheels/* --no-index --find-links=/wheels/ && rm -f *.whl && rm -rf /wheels

# now we can add the application code and install it
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY requirements.txt requirements.txt
COPY litellm/py.typed litellm/py.typed 
COPY litellm  litellm
COPY enterprise  enterprise
RUN pip install -e .
# Generate prisma client
RUN prisma generate
COPY entrypoint.sh  entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 4000/tcp

ENTRYPOINT ["litellm"]

# Append "--detailed_debug" to the end of CMD to view detailed debug logs 
CMD ["--port", "4000"]

