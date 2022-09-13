FROM python:3.8-slim-buster AS builder
WORKDIR /workplace
COPY . .

RUN apt-get clean && \
    apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv .venv &&  \
    .venv/bin/pip install -U pip setuptools && \
    .venv/bin/pip install . --default-timeout=10000 && \
    find /workplace/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM python:3.8-slim
WORKDIR /workplace
COPY --from=builder /workplace /workplace
ENV PATH="/workplace/.venv/bin:$PATH"
COPY apis apis

EXPOSE 8123
CMD ["uvicorn", "apis.interface:app", "--host", "0.0.0.0", "--port", "8123"]