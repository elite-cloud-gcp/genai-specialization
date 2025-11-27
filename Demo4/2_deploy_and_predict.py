# 2_deploy_and_predict.py

from google.cloud import aiplatform
from config import PROJECT_ID, REGION, MODEL_DISPLAY_NAME

def deploy_and_test():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # 1. Find the Model
    models = aiplatform.Model.list(filter=f"display_name={MODEL_DISPLAY_NAME}")
    if not models:
        print("Model not found. Run training first.")
        return
    model = models[0]

    # -------------------------------------------------------
    # EVIDENCE FOR ML 3.4.4.1: Model on Google Cloud (Deployment)
    # -------------------------------------------------------
    print(f"Deploying model {model.resource_name}...")
    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )
    
    print(f"Model Deployed to Endpoint: {endpoint.resource_name}")

    # -------------------------------------------------------
    # EVIDENCE FOR ML 3.4.4.2: Callable Library / Application
    # -------------------------------------------------------
    # Test Data (matches schema expected by pipeline)
    test_instance = [
        {
            "trip_miles": 5.2,
            "trip_seconds": 900,
            "trip_start_hour": 14,
            "trip_day_of_week": 3,
            "payment_type": "Credit Card",
            "company": "Taxi Affiliation Services"
        }
    ]

    print("Sending Prediction Request...")
    prediction = endpoint.predict(instances=test_instance)
    
    print("-----------------------------------")
    print(f"Predicted Fare: {prediction.predictions}")
    print("-----------------------------------")

    # -------------------------------------------------------
    # EVIDENCE FOR ML 3.4.4.3: Editable Model
    # -------------------------------------------------------
    # NOTE: In the whitepaper, you explain that because this is a CI/CD pipeline,
    # updating the 'train_model.py' and re-running 'pipeline_job.py' 
    # automatically updates the deployed model version.

if __name__ == "__main__":
    deploy_and_test()