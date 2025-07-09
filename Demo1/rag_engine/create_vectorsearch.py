import logging

from google import genai
from google.cloud import aiplatform

# pip install --upgrade  google-cloud-aiplatform google-genai

logging.basicConfig(level=logging.INFO)

class VectorSearchManager:
    """
    Manager for Vertex AI Vector Search operations: create index, endpoint, and deploy index.
    """
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        if not self.project_id or not self.location:
            raise ValueError("PROJECT_ID and LOCATION must be provided.")
        aiplatform.init(project=self.project_id, location=self.location)
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)

    def create_streaming_index(self, display_name: str, gcs_uri: str) -> aiplatform.MatchingEngineIndex:
        """
        Create a streaming Matching Engine index.
        """
        try:
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=display_name,
                contents_delta_uri=gcs_uri,
                description="Matching Engine Index",
                dimensions=768,
                approximate_neighbors_count=150,
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=7,
                index_update_method="STREAM_UPDATE",
                distance_measure_type=aiplatform.matching_engine.matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
            )
            logging.info(f"Index created: {index.resource_name}")
            return index
        except Exception as e:
            logging.error(f"Failed to create index: {e}")
            raise

    def create_index_endpoint(self, display_name: str) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Create a Matching Engine index endpoint.
        """
        try:
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=display_name,
                public_endpoint_enabled=True,
                description="Matching Engine Index Endpoint",
            )
            logging.info(f"Index endpoint created: {index_endpoint.name}")
            return index_endpoint
        except Exception as e:
            logging.error(f"Failed to create index endpoint: {e}")
            raise

    def deploy_index(self, index_name: str, index_endpoint_name: str, deployed_index_id: str) -> None:
        """
        Deploy an index to an endpoint.
        """
        try:
            index = aiplatform.MatchingEngineIndex(index_name=index_name)
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
            index_endpoint = index_endpoint.deploy_index(
                index=index, deployed_index_id=deployed_index_id
            )
            logging.info(f"Deployed indexes: {index_endpoint.deployed_indexes}")
        except Exception as e:
            logging.error(f"Failed to deploy index: {e}")
            raise

def main():

    project_id = "a94-project-ai-specialization"
    location = "us-central1"
    
    # 创建VectorSearchManager实例
    manager = VectorSearchManager(project_id=project_id, location=location)
    
    #创建向量搜索索引
    index = manager.create_streaming_index("rag-index", "gs://ragdiydemo/vector_search_index")
    print("向量搜索索引创建成功")

    #创建向量搜索索引端点
    endpoint = manager.create_index_endpoint("rag-endpoint")
    print("向量搜索索引端点创建成功")

    #部署索引
    manager.deploy_index(index.resource_name, endpoint.name, "rag_deployed_index_id")
    print("向量搜索索引部署成功")

if __name__ == "__main__":
    main()
