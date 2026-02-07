# Use Python 3.11 slim image for smaller size (matching .devcontainer)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-compile Python bytecode for faster startup
RUN python -m compileall /usr/local/lib/python3.11

# Copy application code
COPY dashboard_armoedebeleid.py .

# Copy static assets (favicon)
COPY Favicon-alt-2.png .

# Copy Streamlit configuration and static files (fonts)
COPY .streamlit/config.toml .streamlit/
COPY static/ static/

# Pre-compile dashboard bytecode for faster startup
RUN python -m py_compile dashboard_armoedebeleid.py

# Note: Excel file loaded from Dropbox URL at runtime (not bundled in container)
# This allows data updates without rebuilding the container

# Expose port (Cloud Run will override with PORT env var)
EXPOSE 8080

# Run Streamlit with Cloud Run-optimized settings (uses PORT env var from Cloud Run)
CMD ["sh", "-c", "streamlit run dashboard_armoedebeleid.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
