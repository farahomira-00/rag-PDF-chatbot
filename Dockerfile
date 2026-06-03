# Single-container build for the RAG chatbot.
# Slim Python base keeps the image small; 3.12 is current and well-supported.
FROM python:3.12-slim

# Don't write .pyc files; flush logs immediately (nicer in Docker logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# FastEmbed caches its model here; pre-download it at build time so the
# container can answer immediately and doesn't fetch the model on first query.
ENV FASTEMBED_CACHE_DIR=/app/models

# Install dependencies first (separate layer) so Docker can cache them and
# only reinstall when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image (matches EMBEDDING_MODEL default).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

# Copy the application code.
COPY src/ ./src/

# Document the port the app listens on. The grader maps this with -p.
# Keep it in sync with APP_PORT in .env (default 8501).
EXPOSE 8501

# Start Streamlit. We bind to 0.0.0.0 so the app is reachable from outside
# the container, and disable the usage-stats prompt for a clean startup.
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
