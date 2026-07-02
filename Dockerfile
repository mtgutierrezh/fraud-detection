FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .

COPY models/ ./models/

COPY data/ ./data/


RUN mkdir -p /app/logs /app/data/processed /app/reports

EXPOSE 8501

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]