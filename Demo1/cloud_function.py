import base64
import json
import logging
import os
from typing import List
from cloudevents.http import CloudEvent
import functions_framework
from vertexai import rag
import vertexai

# Configure logging
logging.basicConfig(level=logging.INFO)

class RagManager:
    """
    Manager for Vertex AI RAG operations: create corpus, import files, and manage RAG workflows.
    """
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        if not self.project_id or not self.location:
            raise ValueError("PROJECT_ID and LOCATION must be provided.")
        
        # Initialize Vertex AI API
        vertexai.init(project=self.project_id, location=self.location)
        logging.info(f"Initialized Vertex AI with project: {project_id}, location: {location}")

    def import_files_to_corpus(self, corpus_name: str, file_paths: List[str], 
                              chunk_size: int = 1024, chunk_overlap: int = 256):
        """
        Import files to a RAG corpus with chunking configuration.
        """
        try:
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=file_paths,
                transformation_config=rag.TransformationConfig(
                    rag.ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                )
            )
            logging.info(f"Imported {response.imported_rag_files_count} files to corpus.")
            return response
        except Exception as e:
            logging.error(f"Failed to import files: {e}")
            raise

# Get configuration from environment variables
PROJECT_ID = os.getenv("PROJECT_ID", "a94-project-ai-specialization")
LOCATION = os.getenv("LOCATION", "us-central1")
CORPUS_NAME = os.getenv("CORPUS_NAME", "projects/41449521287/locations/us-central1/corpora/1234567890123456789")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Initialize RagManager
rag_manager = RagManager(project_id=PROJECT_ID, location=LOCATION)

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    """
    Cloud Function triggered by Pub/Sub message to import files to RAG corpus.
    """
    data = cloud_event.data["message"]["data"]
    try:
        message_data = base64.b64decode(data)
        message = json.loads(message_data)

        filename = message["name"]
        bucket = message["bucket"]
    except Exception as e:
        raise ValueError(f"Missing or malformed PubSub message {data}: {e}.")

    print(f"Processing file: {filename} from bucket: {bucket}")
    gcs_file_uri = f"gs://{bucket}/{filename}"
    
    try:
        # Import file to RAG corpus
        print(f"Importing file to RAG corpus: {gcs_file_uri}")
        print(f"Using corpus: {CORPUS_NAME}")
        print(f"Chunk size: {CHUNK_SIZE}, Chunk overlap: {CHUNK_OVERLAP}")
        
        response = rag_manager.import_files_to_corpus(
            corpus_name=CORPUS_NAME,
            file_paths=[gcs_file_uri],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        print(f"Successfully imported file to RAG corpus. Imported {response.imported_rag_files_count} files.")
        return {
            "status": "success",
            "message": f"File {filename} imported to RAG corpus successfully",
            "imported_files_count": response.imported_rag_files_count,
            "corpus_name": CORPUS_NAME,
            "file_uri": gcs_file_uri
        }
        
    except Exception as e:
        error_msg = f"Failed to import file {filename} to RAG corpus: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        raise Exception(error_msg)