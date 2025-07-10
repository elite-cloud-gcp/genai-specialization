from vertexai import rag
from vertexai.generative_models import SafetySetting,GenerativeModel, Tool
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

def rag_gemini_tool_query(query_text: str, system_prompt: str, generation_config,safety_settings):
    """
    使用 rag_retrieval_tool 方法进行查询，支持 system prompt
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
        model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool],system_instruction=system_prompt
    )
    response = rag_model.generate_content(
        query_text,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False
    )
    print("=== 查询结果 ===")
    print(response.text)
    return response

# 示例使用
if __name__ == "__main__":
    
    system_prompt = "You are a helpful assistant who can answer questions about the products in the corpus and other assorted questions."
    contents = "I want to buy Vishudh brand products, what products can you recommend and are also cheap?" 

    generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
    "seed": 0,
}

    safety_settings=[
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

    #retrieval_query(query)

    rag_gemini_tool_query(contents, system_prompt, generation_config,safety_settings)