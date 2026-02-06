# Use a slim Python 3.12 base image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (Node.js and npx are required for the MCP server)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*



# Install 'uv' for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
# --frozen ensures we use the exact versions in the lockfile
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for LangGraph server (default is 8000)
EXPOSE 2024

# Start the LangGraph development server
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]
