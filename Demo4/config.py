# config.py
import os

# PROJECT SETTINGS
PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
BUCKET_URI = "gs://your-bucket-name"

# DATA SETTINGS
# Using Google Cloud Public Dataset
BQ_SOURCE_TABLE = "bigquery-public-data.chicago_taxi_trips.taxi_trips"
BQ_DATASET_ID = "taxi_ml_dataset"  # Your working dataset
BQ_DEST_TABLE = "training_data"

# MODEL SETTINGS
MODEL_DISPLAY_NAME = "chicago-taxi-fare-predictor"
SERVING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"
TRAINING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-6:latest"