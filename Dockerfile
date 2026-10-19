# ------------- Builder Stage -------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app

# Copy dependency files only
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (locked, reproducible)
RUN uv sync --frozen --no-dev

# ------------- Runtime Stage -------------
FROM python:3.12-slim AS runtime

# Install minimal runtime packages
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Runtime environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_HOST=0.0.0.0 \
    PORT=8080 \
    CONTEXT=/ \
    OUTPUT_DIR=/app/output \
    REPORT_OUTPUT_FILE=report.md

# Create restricted non-root user
RUN groupadd --system --gid 1001 calibo \
    && useradd --system --uid 1001 --gid calibo \
    --home-dir /app --no-create-home \
    --shell /usr/sbin/nologin calibo

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Ensure venv is executable
RUN chmod -R 755 /app/.venv

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=calibo:calibo src ./src

# Create writable directories required by the application
RUN mkdir -p /app/output /app/.local /app/.cache \
    && chown -R calibo:calibo /app/output /app/.local /app/.cache

# Restrictive permissions
RUN chmod 755 /app \
    && chmod -R 755 /app/src \
    && chmod -R 755 /app/output \
    && chmod -R 755 /app/.venv \
    && chmod -R 755 /app/.local \
    && chmod -R 755 /app/.cache \
    && find /app/src -name "*.py" -exec chmod 644 {} \;

# Cleanup
RUN apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/cache/apt/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Run as non-root user
USER calibo

ENTRYPOINT ["tini", "--"]

# Keep the existing start command contract
CMD ["python", "-m", "latest_ai_development.main"]
