FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

CMD ["sh", "-c", "python run_pipeline.py && python app.py"]
