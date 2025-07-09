from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

PROJECT_ID = "a94-project-ai-specialization"
corpus_name = "projects/a94-project-ai-specialization/locations/us-central1/ragCorpora/3379951520341557248"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location="us-central1")

def retrieval_query(query_text: str):
    """
    使用 rag.retrieval_query 方法进行查询
    """
    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=corpus_name,
            )
        ],
        text=query_text,
        rag_retrieval_config=rag.RagRetrievalConfig(
            top_k=5,
            filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
        ),
    )
    print("=== 查询结果 ===")
    print(response)
    return response

def rag_gemini_tool_query(query_text: str):
    """
    使用 rag_retrieval_tool 方法进行查询
    """
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=corpus_name,
                    )
                ],
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=5,
                    filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
                ),
            ),
        )
    )

    rag_model = GenerativeModel(
        model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
    )
    response = rag_model.generate_content(query_text)
    print("=== 查询结果 ===")
    print(response.text)
    return response

# 示例使用
if __name__ == "__main__":

    query="I want to buy Vishudh brand products, what products can you recommend and are also cheap?"
        
    retrieval_query(query)

    rag_gemini_tool_query(query)