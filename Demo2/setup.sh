#!/bin/bash

# Project setup - Replace placeholders with your project's details
PROJECT_ID="hostproject1-355311"         # Your Google Cloud Project ID
LOCATION="us-central1"               # Google Cloud region (e.g., us-central1)
ARTIFACT_REGISTRY_REPO="customer-manager-repo"  # Name for your Docker image repository
SERVICE_NAME="customer-api"         # Name for your Cloud Run service
BUILD_CONTEXT="./customer-manager-web"  # Specify the directory containing your Dockerfile and source code

# Activate necessary Google Cloud services for building and running
echo "***** Enabling Cloud Build and Cloud Run *****"
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Create a repository to store the Docker image of the web application
echo "***** Creating Artifact Repository *****"
gcloud artifacts repositories create "$ARTIFACT_REGISTRY_REPO" --location="$LOCATION" --repository-format=Docker > /dev/null
echo "***** Repository created *****"

# Set up authentication to interact with the Docker repository
echo "***** Setting up Docker authentication *****"
gcloud auth configure-docker "$LOCATION-docker.pkg.dev" --quiet > /dev/null

# Build the Docker image containing the web application code
echo "***** Building Docker image *****"
gcloud builds submit --tag "$LOCATION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/$SERVICE_NAME" "$BUILD_CONTEXT" > /dev/null

# Deploy the Docker image to Cloud Run, making it accessible
echo "***** Deploying to Cloud Run *****"
gcloud run deploy "$SERVICE_NAME" --port=8080 --image="$LOCATION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/$SERVICE_NAME" --allow-unauthenticated --region=$LOCATION --platform=managed --project=$PROJECT_ID --set-env-vars=PROJECT_ID=$PROJECT_ID,LOCATION=$LOCATION > /dev/null

# Allow anyone to access the Cloud Run service (open access)
echo "***** Setting Cloud Run service to allow public access *****"
gcloud run services add-iam-policy-binding "$SERVICE_NAME" --member="allUsers" --role="roles/run.invoker" --region="$LOCATION" > /dev/null

# Display the web address (URL) of the deployed Cloud Run service
echo "***** Cloud RUN URL *****"
APP_URL=$(gcloud run services describe $SERVICE_NAME --region=us-central1 --format="value(status.url)")
echo $APP_URL
