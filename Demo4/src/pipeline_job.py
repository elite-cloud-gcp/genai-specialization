# src/pipeline_job.py

from google.cloud import aiplatform
from config import (PROJECT_ID, REGION, BUCKET_URI, TRAINING_IMAGE_URI, 
                    SERVING_IMAGE_URI, MODEL_DISPLAY_NAME, BQ_DEST_TABLE)

def run_training_job():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # -------------------------------------------------------
    # EVIDENCE FOR ML 3.4.3.6: Implementation of Model Training on GCP
    # -------------------------------------------------------
    job = aiplatform.CustomTrainingJob(
        display_name=f"{MODEL_DISPLAY_NAME}-job",
        script_path="src/train_model.py",     # The script defined above
        container_uri=TRAINING_IMAGE_URI,
        requirements=["google-cloud-bigquery", "scikit-learn", "pandas", "pyarrow"],
        model_serving_container_image_uri=SERVING_IMAGE_URI
    )

    print("Submitting Training Job...")
    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
        args=[
            f"--project_id={PROJECT_ID}",
            f"--bq_table={PROJECT_ID}.taxi_ml_dataset.{BQ_DEST_TABLE}",
        ],
        replica_count=1,
        machine_type="n1-standard-4",
    )
    
    return model

if __name__ == "__main__":
    run_training_job()