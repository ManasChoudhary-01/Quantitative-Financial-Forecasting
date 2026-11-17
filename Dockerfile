# Container image for the FastAPI backend (the API service).
# The Streamlit UI is typically deployed separately (e.g. Streamlit Cloud),
# pointed at this API via the API_URL env var.

FROM python:3.11-slim

# Keep Python output unbuffered and skip .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# The API listens on $PORT (Render/Railway set this) or 8000 locally.
ENV PORT=8000
EXPOSE 8000

# NOTE: run `python -m ingestion.ingest` once (locally or as a release step)
# to build the index before/at deploy. With VECTOR_BACKEND=pinecone the index
# lives in Pinecone, so the container itself stays stateless.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
