FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /workplace
COPY . .
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2 && \
    pip install -U pip && \
    pip install . && \
    pip install --upgrade Pillow && \
    rm -rf ./*

EXPOSE 8123
CMD ["cfcreator", "serve"]