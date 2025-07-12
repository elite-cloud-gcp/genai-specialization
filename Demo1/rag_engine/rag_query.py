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
    
    system_prompt = f"""
    你是专业的电子商务产品比较专家。
    请根据以下提供的两个产品的信息，详细比较它们之间在用户关注点上的异同。
    最终，请总结它们的优缺点，并给出一个购买建议。
       
    请按以下步骤思考并回答：
    1.  **产品特点提取：** 从<产品A信息>和<产品B信息>中分别提取与“用户关注点”直接相关的关键特点。
    2.  **逐点对比：** 针对每个关键特点，对比产品A和产品B的异同。
    3.  **优缺点总结：** 根据对比结果，总结产品A和产品B的各自优缺点。
    4.  **购买建议：** 结合所有分析，为用户提供一个基于其关注点的个性化购买建议。
    
    请直接给出最终的比较结果和建议，格式清晰。
    """
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