# Use Python 3.11 slim image for smaller size (matching .devcontainer)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by openpyxl and pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dashboard_armoedebeleid.py .

# Copy static assets (favicon and logo)
COPY Favicon-alt-2.png .
COPY "IPE Logo 01.png" .

# Copy Streamlit configuration
COPY .streamlit/config.toml .streamlit/

# Note: Excel file loaded from Dropbox URL at runtime (not bundled in container)
# This allows data updates without rebuilding the container

# Expose Streamlit's default port
EXPOSE 8501

# Health check for Cloud Run monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit with Cloud Run-optimized settings
CMD ["streamlit", "run", "dashboard_armoedebeleid.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
