FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_hf.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy backend code
COPY backend/ ./backend/
COPY src/ ./src/
COPY config.yaml .

# Copy frontend build
COPY frontend/dist/ ./frontend/dist/

# Copy main app
COPY app.py .

# Create models directory
RUN mkdir -p backend/models

# Copy model files (you'll need to add these)
# The model file should be committed with Git LFS
COPY backend/models/plant_disease_model.pth backend/models/
COPY backend/models/class_names.json backend/models/

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Run the application
CMD ["python", "app.py"]
