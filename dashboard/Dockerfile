ARG NODE_BASE_IMAGE=node:22-bookworm-slim
ARG PYTHON_BASE_IMAGE=python:3.11-slim-bookworm
ARG NO_MIRROR

# Set NO_MIRROR=1 to keep the upstream Debian and PyPI sources.
FROM ${NODE_BASE_IMAGE} AS node-base
ARG NO_MIRROR
ENV NO_MIRROR=${NO_MIRROR}
RUN if [ "$NO_MIRROR" != "1" ]; then \
        for source_file in \
            /etc/apt/sources.list \
            /etc/apt/sources.list.d/debian.sources; do \
            if [ -f "$source_file" ]; then \
                sed -i \
                    -e 's@//deb.debian.org@//mirrors.ivolces.com@g' \
                    -e 's@//security.debian.org@//mirrors.ivolces.com@g' \
                    "$source_file"; \
            fi; \
        done; \
    fi

FROM node-base AS frontend-build
WORKDIR /src
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json ./frontend/
RUN --mount=type=cache,target=/root/.npm \
    npm --prefix frontend ci
COPY dashboard/frontend ./frontend
RUN npm --prefix frontend run build

FROM ${PYTHON_BASE_IMAGE} AS python-base
ARG NO_MIRROR
ENV NO_MIRROR=${NO_MIRROR}
ENV PIP_INDEX_URL=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV PIP_INDEX_URL=${PIP_INDEX_URL:-"https://mirrors.ivolces.com/pypi/simple/"}
RUN if [ "$NO_MIRROR" != "1" ]; then \
        for source_file in \
            /etc/apt/sources.list \
            /etc/apt/sources.list.d/debian.sources; do \
            if [ -f "$source_file" ]; then \
                sed -i \
                    -e 's@//deb.debian.org@//mirrors.ivolces.com@g' \
                    -e 's@//security.debian.org@//mirrors.ivolces.com@g' \
                    "$source_file"; \
            fi; \
        done; \
    fi

FROM python-base AS wheel-build
WORKDIR /src
COPY dashboard/ .
COPY --from=frontend-build /src/frontend/dist ./frontend/dist
RUN python scripts/bundle_frontend.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --wheel-dir /wheels .

FROM python-base AS runtime
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
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8420/healthz', timeout=2)"]

ENTRYPOINT ["rlinf-dashboard"]
CMD ["/runs", "--host", "0.0.0.0", "--port", "8420"]
