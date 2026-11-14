FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/app/weights DB_PATH=/data/events.db FEAST_REPO=/app/feature_repo
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY fraud ./fraud
COPY serving ./serving
COPY training ./training
COPY monitoring ./monitoring
COPY dashboard ./dashboard
COPY feature_repo ./feature_repo
COPY weights ./weights
COPY data/replay_sample.csv ./data/replay_sample.csv
RUN mkdir -p /data && chmod +x serving/start_api.sh
CMD ["serving/start_api.sh"]
