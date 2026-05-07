# syntax=docker/dockerfile:1
#
# LiveKit Cloud build for the Vilnius savivaldybė Soniox-TTS agent.
# Adapted from the Bite Lietuva demo Dockerfile.

ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# --- Build stage --------------------------------------------------------
FROM base AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ python3-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Resolve deps first so the layer is cached across code edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Copy the rest of the project (filtered by .dockerignore).
COPY . .

# Pre-download Silero VAD weights so cold-start is fast.
RUN uv run agent.py download-files

# --- Production stage --------------------------------------------------
FROM base

ARG UID=10001
RUN adduser --disabled-password --gecos "" --home /app --shell /sbin/nologin \
        --uid "${UID}" appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app

USER appuser

CMD ["uv", "run", "agent.py", "start"]
