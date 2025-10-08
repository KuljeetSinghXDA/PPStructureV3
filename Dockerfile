FROM arm64v8/ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake git python3-full python3-dev swig wget patchelf libopencv-dev \
    libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev gfortran libpng-dev libfreetype6-dev libjpeg-dev zlib1g-dev \
    libnss-systemd \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV PYTHON_VENV_PATH=/opt/venv
RUN python3 -m venv ${PYTHON_VENV_PATH}
ENV PATH=${PYTHON_VENV_PATH}/bin:$PATH

# Upgrade pip and install protobuf (pinned for compatibility)
RUN pip install --upgrade pip setuptools wheel
RUN pip install protobuf==3.20.3

# Clone Paddle with all submodules
RUN git clone --recurse-submodules https://github.com/PaddlePaddle/Paddle.git /Paddle && \
    cd /Paddle && \
    git checkout v3.2.0 && \
    git submodule sync --recursive && \
    git submodule update --init --recursive

# Safely disable -Werror to avoid ARM64 build failures from warnings
RUN cd /Paddle && sed -i 's/-Werror=/-W/g' cmake/flags.cmake

# Install build-time Python dependencies
RUN cd /Paddle && pip install -r python/requirements.txt

# Build Paddle (ARM64 CPU-only)
RUN mkdir -p /Paddle/build && cd /Paddle/build && \
    cmake .. \
    -DPY_VERSION=3.12 \
    -DWITH_GPU=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON \
    -DWITH_ARM=ON \
    -DARM_TARGET_ARCH=armv8 \
    -DARM_TARGET_LANG=gcc \
    -DWITH_MKL=OFF \
    -DWITH_MKLDNN=OFF \
    -DWITH_AVX=OFF \
    -DWITH_XBYAK=OFF \
    -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    -DPYTHON_INCLUDE_DIR=/opt/venv/include/python3.12 \
    -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.12.so && \
    ulimit -n 8192 && \
    make TARGET=ARMV8 -j2 && \
    pip install python/dist/*.whl

# Install PaddleOCR
RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git /PaddleOCR && \
    cd /PaddleOCR && \
    git checkout release/3.2

RUN cd /PaddleOCR && pip install -r requirements.txt && pip install paddleocr==3.2.0 paddlehub==2.4.0

# Download inference models
RUN mkdir -p /PaddleOCR/inference

# PP-OCRv5 detection
RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar -O /PaddleOCR/inference/PP-OCRv5_server_det_infer.tar && \
    tar -xf /PaddleOCR/inference/PP-OCRv5_server_det_infer.tar -C /PaddleOCR/inference && \
    rm /PaddleOCR/inference/PP-OCRv5_server_det_infer.tar

# Classification model
RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar -O /PaddleOCR/inference/PP-LCNet_x1_0_textline_ori_infer.tar && \
    tar -xf /PaddleOCR/inference/PP-LCNet_x1_0_textline_ori_infer.tar -C /PaddleOCR/inference && \
    rm /PaddleOCR/inference/PP-LCNet_x1_0_textline_ori_infer.tar

# PP-OCRv5 recognition
RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar -O /PaddleOCR/inference/PP-OCRv5_server_rec_infer.tar && \
    tar -xf /PaddleOCR/inference/PP-OCRv5_server_rec_infer.tar -C /PaddleOCR/inference && \
    rm /PaddleOCR/inference/PP-OCRv5_server_rec_infer.tar

# Update model paths in params.py
RUN sed -i 's|det_model_dir = .*|det_model_dir = "./inference/PP-OCRv5_server_det_infer/"|g' /PaddleOCR/deploy/hubserving/ocr_system/params.py && \
    sed -i 's|cls_model_dir = .*|cls_model_dir = "./inference/PP-LCNet_x1_0_textline_ori_infer/"|g' /PaddleOCR/deploy/hubserving/ocr_system/params.py && \
    sed -i 's|rec_model_dir = .*|rec_model_dir = "./inference/PP-OCRv5_server_rec_infer/"|g' /PaddleOCR/deploy/hubserving/ocr_system/params.py

WORKDIR /PaddleOCR

RUN hub install deploy/hubserving/ocr_system/

EXPOSE 8868

CMD ["hub", "serving", "start", "-m", "ocr_system", "-p", "8868"]
