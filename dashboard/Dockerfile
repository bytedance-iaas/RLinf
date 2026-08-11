# syntax=docker/dockerfile:1.7

ARG NODE_BASE_IMAGE=node:22-bookworm-slim
ARG PYTHON_BASE_IMAGE=python:3.11-slim-bookworm

FROM ${NODE_BASE_IMAGE} AS frontend-build
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN --mount=type=cache,target=/root/.npm \
    npm --prefix frontend ci
COPY frontend ./frontend
RUN npm --prefix frontend run build

FROM ${PYTHON_BASE_IMAGE} AS wheel-build
WORKDIR /src
COPY . .
COPY --from=frontend-build /src/frontend/dist ./frontend/dist
RUN python scripts/bundle_frontend.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --wheel-dir /wheels .

FROM ${PYTHON_BASE_IMAGE} AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=bind,from=wheel-build,source=/wheels,target=/wheels \
    python -m pip install --no-cache-dir --no-index \
      --find-links=/wheels rlinf-dashboard \
    && useradd --create-home --uid 10001 dashboard

USER dashboard
WORKDIR /app
VOLUME ["/runs"]
EXPOSE 8420

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8420/api/health', timeout=2)"]

ENTRYPOINT ["rlinf-dashboard"]
CMD ["/runs", "--host", "0.0.0.0", "--port", "8420"]
