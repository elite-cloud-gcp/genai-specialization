import logging
from typing import List

from vertexai import rag
import vertexai

# pip install --upgrade google-cloud-aiplatform google-genai

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

    def create_corpus(self, display_name: str, description: str, 
                     vector_search_index_name: str, 
                     vector_search_index_endpoint_name: str,
                     embedding_model: str = "publishers/google/models/text-embedding-005") -> rag.RagCorpus:
        """
        Create a RAG corpus with vector search configuration.
        """
        try:
            # Configure embedding model
            embedding_model_config = rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=embedding_model
                )
            )

            # Configure Vector DB
            vector_db = rag.VertexVectorSearch(
                index=vector_search_index_name, 
                index_endpoint=vector_search_index_endpoint_name
            )

            # Create corpus
            corpus = rag.create_corpus(
                display_name=display_name,
                description=description,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config,
                    vector_db=vector_db,
                ),
            )
            logging.info(f"Corpus created: {corpus}")
            return corpus
        except Exception as e:
            logging.error(f"Failed to create corpus: {e}")
            raise

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


def create_rag_corpus():
    """
    创建RAG语料库的独立函数
    """
    # 手动输入PROJECT_ID和LOCATION
    project_id = "a94-project-ai-specialization"
    location = "us-central1"
    
    # 创建RagManager实例
    manager = RagManager(project_id=project_id, location=location)
    
    # 配置参数
    vector_search_index_name = "projects/41449521287/locations/us-central1/indexes/8727151444123189248"
    vector_search_index_endpoint_name = "projects/41449521287/locations/us-central1/indexEndpoints/6411333665422311424"
    display_name = "rag_corpus"
    description = "RAG Corpus Description"
    
    # 创建RAG语料库
    corpus = manager.create_corpus(
        display_name=display_name,
        description=description,
        vector_search_index_name=vector_search_index_name,
        vector_search_index_endpoint_name=vector_search_index_endpoint_name
    )
    print(f"RAG语料库创建成功: {corpus}")
    return corpus, manager

def import_files_to_rag_corpus(corpus_name: str):
    """
    导入文件到RAG语料库的独立函数
    """
    # 手动输入PROJECT_ID和LOCATION
    project_id = "a94-project-ai-specialization"
    location = "us-central1"
    
    # 创建RagManager实例
    manager = RagManager(project_id=project_id, location=location)
    
    # 输入文件路径
    file_paths = ["gs://ragdiydemo/ecommerce_sample"]
    
    # 输入分块配置
    chunk_size = 512
    chunk_overlap = 100
    
    # 导入文件
    response = manager.import_files_to_corpus(
        corpus_name=corpus_name,
        file_paths=file_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"文件导入成功，共导入 {response.imported_rag_files_count} 个文件")
    return response

def main():
    try:
        # 步骤1: 创建RAG语料库
        print("\n--- 步骤1: 创建RAG语料库 ---")
        corpus, manager = create_rag_corpus()
        print(f"语料库创建完成，名称: {corpus.name}")
        
        # 步骤2: 导入文件到语料库
        print("\n--- 步骤2: 导入文件到语料库 ---")
        response = import_files_to_rag_corpus(corpus.name)
        print(f"文件导入成功，共导入 {response.imported_rag_files_count} 个文件")
        
        print("\n=== 所有操作完成 ===")
        
    except Exception as e:
        print(f"操作失败: {e}")

if __name__ == "__main__":
    main()