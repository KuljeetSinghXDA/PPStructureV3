FROM arm64v8/ubuntu:24.04

RUN apt-get update && apt-get install -y \
    gcc g++ make cmake git python3-dev python3-pip swig wget patchelf libopencv-dev \
    libatlas-base-dev libopenblas-dev libblas-dev liblapack-dev gfortran libpng-dev libfreetype6-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip --break-system-packages && \
    python3 -m pip install --break-system-packages protobuf==3.20.3

RUN git clone https://github.com/PaddlePaddle/Paddle.git /Paddle && cd /Paddle && git checkout v3.2.0

RUN cd /Paddle && python3 -m pip install --break-system-packages -r python/requirements.txt

RUN mkdir /Paddle/build && cd /Paddle/build && cmake .. \
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
    && make TARGET=ARMV8 -j4 && python3 -m pip install --break-system-packages python/dist/*.whl

RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git /PaddleOCR && cd /PaddleOCR && git checkout release/3.2

RUN cd /PaddleOCR && python3 -m pip install --break-system-packages -r requirements.txt && python3 -m pip install --break-system-packages paddleocr==3.2.0 paddlehub==2.4.0

RUN mkdir -p /PaddleOCR/inference

# Download PP-OCRv5 detection model (server version for high accuracy)
RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar -O /PaddleOCR/inference/PP-OCRv5_server_det_infer.tar && \
    tar -xf /PaddleOCR/inference/PP-OCRv5_server_det_infer.tar -C /PaddleOCR/inference && rm /PaddleOCR/inference/PP-OCRv5_server_det_infer.tar

# Download latest classification model (PP-LCNet_x1_0_textline_ori for 99.42% accuracy)
RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar -O /PaddleOCR/inference/PP-LCNet_x1_0_textline_ori_infer.tar && \
    tar -xf /PaddleOCR/inference/PP-LCNet_x1_0_textline_ori_infer.tar -C /PaddleOCR/inference && rm /PaddleOCR/inference/PP-LCNet_x1_0_textline_ori_infer.tar

# Download PP-OCRv5 recognition model (server version for high accuracy)
RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar -O /PaddleOCR/inference/PP-OCRv5_server_rec_infer.tar && \
    tar -xf /PaddleOCR/inference/PP-OCRv5_server_rec_infer.tar -C /PaddleOCR/inference && rm /PaddleOCR/inference/PP-OCRv5_server_rec_infer.tar

# Update params.py to point to the latest models
RUN sed -i 's|det_model_dir = .*|det_model_dir = "./inference/PP-OCRv5_server_det_infer/"|g' /PaddleOCR/deploy/hubserving/ocr_system/params.py && \
    sed -i 's|cls_model_dir = .*|cls_model_dir = "./inference/PP-LCNet_x1_0_textline_ori_infer/"|g' /PaddleOCR/deploy/hubserving/ocr_system/params.py && \
    sed -i 's|rec_model_dir = .*|rec_model_dir = "./inference/PP-OCRv5_server_rec_infer/"|g' /PaddleOCR/deploy/hubserving/ocr_system/params.py

WORKDIR /PaddleOCR

RUN hub install deploy/hubserving/ocr_system/

EXPOSE 8868

CMD ["hub", "serving", "start", "-m", "ocr_system", "-p", "8868"]
