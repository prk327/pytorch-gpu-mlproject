# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:22.07-py3

LABEL maintainer="Prabhakar Srivastav <prk327@gmail.com>"

# -- Layer: Image Metadata

LABEL org.label-schema.name="Data Science Standalone Cluster on Docker - Cluster Base Image"
LABEL org.label-schema.description="Cluster base image shipped with Python , jupyter lab and pytorch"
LABEL org.label-schema.url="https://github.com/prk327/pytorch-gpu-mlproject.git"
LABEL org.label-schema.schema-version="1.0"

RUN apt update && apt install --only-upgrade linux-libc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# create a password for root
RUN echo 'root:${rtpass}' | chpasswd

# Create a non-root user
ARG username
ARG uidr
ARG gidr
ENV HOMEN /home/${username}

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid ${uidr} \
    --gid ${gidr} \
    --home $HOMEN \
    $username

# running the below code as non root user
USER ${username}

# default command will be to launch JupyterLab server for development
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
