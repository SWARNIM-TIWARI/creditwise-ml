# ------------------------------
# 1. Base image: Python 3.10 slim
# ------------------------------
FROM python:3.10-slim

# ------------------------------
# 2. Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# 3. Copy project files
# ------------------------------
COPY . /app

# ------------------------------
# 4. Install system dependencies
# ------------------------------
RUN apt-get update && \
    apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------
# 5. Install Python dependencies
# ------------------------------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ------------------------------
# 6. Expose port for Gradio
# ------------------------------
EXPOSE 7860

# ------------------------------
# 7. Launch the app
# ------------------------------
CMD ["python", "run_pipeline.py"]
