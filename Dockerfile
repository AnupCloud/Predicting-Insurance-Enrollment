FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app

WORKDIR /app
RUN uv add -r requirements.txt
RUN uv sync --frozen --no-cache

CMD ["/app/.venv/bin/uvicorn", "main:app", "--port", "8000", "--host","0.0.0.0", "--port", "8000"]