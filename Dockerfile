FROM nvcr.io/nvidia/pytorch:22.09-py3
WORKDIR /workplace
COPY . .
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2 && \
    pip install -U pip && \
    pip install . && \
    pip install --upgrade Pillow && \
    rm -rf ./*

EXPOSE 8123
CMD ["cfcreator", "serve"]