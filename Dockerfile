ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install necessary packages
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages and dependencies
RUN pip install -U openmim gdown python-telegram-bot Pillow
RUN mim install mmengine
RUN mim install 'mmcv>=2.0.0rc4,<2.2.0'
RUN pip install 'mmdet>=3.0.0rc0'
RUN git clone https://github.com/open-mmlab/mmocr.git /mmocr
WORKDIR /mmocr

# Set environment variable to force CUDA usage
ENV FORCE_CUDA="1"

# Install more dependencies and gradio
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .
RUN pip install -r requirements/albu.txt

# Copy all project files to the working directory
COPY . /mmocr/

# Default command to run when the container starts
CMD ["python3", "app.py"]
