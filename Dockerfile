# One image for both roles: the mock chatbot server and the quality-gate runner.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/artifacts \
    && chown -R app:app /app
USER app

EXPOSE 8080
CMD ["uvicorn", "mock_chatbot.app:app", "--host", "0.0.0.0", "--port", "8080"]
