FROM python:3.14-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ ./src/
COPY main.py ./main.py

RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir .


FROM python:3.14-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY main.py .
COPY src/ ./src/

EXPOSE 8000

USER app

ENTRYPOINT ["python3", "main.py"]
CMD ["-c", "/app/config.yml"]