FROM python:3.11-slim

# Install system deps required by Playwright and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency manifest first (layer cache)
COPY pyproject.toml .

# Install dependencies (no project, no dev extras)
RUN uv sync --no-dev

# Install Playwright browsers
RUN uv run playwright install --with-deps chromium

# Copy source
COPY app.py .
COPY src/ src/

# Streamlit listens on 8501 by default
EXPOSE 8501

# Disable Streamlit telemetry and browser auto-open
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
