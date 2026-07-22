FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app/ ./app/
COPY data/raw/ ./data/raw/

RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "app.scripts.search_demo"]
