# src/train_model.py

import argparse
import logging
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery
from google.cloud import storage

# best practice: Use argparse to allow Vertex AI to inject parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir', default='gs://my-bucket/model_output', type=str, help='GCS location to save model')
parser.add_argument('--n_estimators', dest='n_estimators', default=100, type=int)
parser.add_argument('--learning_rate', dest='learning_rate', default=0.1, type=float)
args = parser.parse_args()

# 1. Ingestion (Corrected for Random Sampling)
# Rationale: LIMIT is not random. We use RAND() < 0.X to statistically sample approx 50k rows
# assuming total dataset is large, 0.01% might be needed. 
# Alternatively, use HASH for repeatable sampling: MOD(ABS(FARM_FINGERPRINT(unique_key)), 100) = 1
client = bigquery.Client()
query = """
    SELECT * FROM `a94-project-ai-specialization.taxi_ml_dataset.clean_training_data`
    WHERE RAND() < 0.05 
    LIMIT 50000
"""
df = client.query(query).to_dataframe()

# 2. Splitting (Bias/Variance Check)
# Strategy: 80% Train, 20% Test for Generalization measurement
X = df.drop('fare', axis=1)
y = df['fare']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline Construction
# Note: 'preprocessor' must be defined or imported from a utility module
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=args.n_estimators,   # Passed via Vertex AI
        learning_rate=args.learning_rate, # Passed via Vertex AI
        max_depth=5,
        n_jobs=-1 # Use all CPUs available in the container
    ))
])

# 4. Training
logging.info("Starting training...")
model_pipeline.fit(X_train, y_train)

# 5. Evaluation & Metric Logging
predictions = model_pipeline.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
logging.info(f"Validation RMSE: {rmse}")

# 6. Artifact Saving (Google Cloud Best Practice)
# Save locally to container, then upload to GCS
local_file = 'model.pkl'
with open(local_file, 'wb') as f:
    pickle.dump(model_pipeline, f)

# Upload to the GCS path provided by Vertex AI
bucket_name = args.model_dir.replace('gs://', '').split('/')[0]
blob_path = '/'.join(args.model_dir.replace('gs://', '').split('/')[1:]) + '/model.pkl'

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_path)
blob.upload_from_filename(local_file)

print(f"Model saved to {args.model_dir}")
