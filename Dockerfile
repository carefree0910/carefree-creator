FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2 && \
    pip install -U pip && \
    apt-get update && \
    apt-get install -y gcc git tzdata && \
    pip install --upgrade Pillow && \
    apt-get clean && \
    rm -rf /tmp/* /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/* && \
COPY . .
RUN pip install . && rm -rf ./*

EXPOSE 8123

CMD ["cfcreator", "serve"]