FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

RUN apt-get -y install git

# install python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

COPY ./requirements/prod.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r prod.txt

# copy files
WORKDIR /app/
ENV PYTHONPATH=/app
COPY ./ /app/

# start server
EXPOSE 5000
CMD ["python", "inference_server/app.py"]