import json
from typing import Dict, Any
from vertexai.generative_models import GenerativeModel
import sys
sys.path.append("../rag_engine")
from rag_query import retrieval_query, rag_gemini_tool_query, corpus_name

def evaluate_answer_factuality_with_llm(
    query: str, 
    generated_answer: str, 
    retrieved_context: str, 
    llm_evaluator_model: GenerativeModel
) -> Dict[str, Any]:
    """
    使用另一个LLM评估生成答案的事实准确性。
    """
    evaluation_prompt = f"""
    你是一个严谨的事实核查专家。
    请判断以下“生成答案”是否完全基于提供的“检索上下文”，并且准确地回答了“用户查询”。
    
    用户查询: {query}
    
    检索上下文:
    {retrieved_context}
    
    生成答案:
    {generated_answer}
    
    请根据以下几点进行评估：
    1.  **事实忠实度 (Fidelity)：** 生成答案中的所有信息是否都可以直接从“检索上下文”中找到或合理推断？(是/否)
    2.  **相关性 (Relevance)：** 生成答案是否完全解决了“用户查询”中的问题？(是/否)
    3.  **幻觉 (Hallucination)：** 生成答案中是否存在“检索上下文”中不存在的，或与“检索上下文”冲突的信息？(是/否)

    请以JSON格式返回评估结果，例如：
    {{
      "fidelity": "是",
      "relevance": "是",
      "hallucination": "否",
      "reasoning": "评估理由的详细说明，例如：生成答案中的每一个信息点都对应于检索上下文中的某一段，并且回答了查询中的核心问题。"
    }}
    """
    try:
        response = llm_evaluator_model.generate_content(evaluation_prompt)
        eval_result_json = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        return eval_result_json
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return {"error": str(e), "fidelity": "未知", "relevance": "未知", "hallucination": "未知"}

if __name__ == "__main__":
    # 查询内容
    query = "I want to buy Vishudh brand products, what products can you recommend and are also cheap?"
    # 1. 获取检索上下文
    retrieval_response = retrieval_query(query)
    print(type(retrieval_response.contexts))
    print(dir(retrieval_response.contexts))
    # 提取所有检索到的上下文文本
    if hasattr(retrieval_response, 'contexts'):
        # RagContexts对象的contexts属性才是真正的list
        contexts_list = retrieval_response.contexts.contexts
        retrieved_context = "\n".join([c.text for c in contexts_list])
    elif isinstance(retrieval_response, dict) and 'contexts' in retrieval_response:
        retrieved_context = "\n".join([c['text'] for c in retrieval_response['contexts']])
    else:
        retrieved_context = str(retrieval_response)
    print("\n=== 检索上下文 ===\n", retrieved_context)

    # 2. 用RAG生成答案
    system_prompt = "You are a helpful assistant who can answer questions about the products in the corpus and other assorted questions."
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
        "seed": 0,
    }
    from vertexai.generative_models import SafetySetting
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
    rag_response = rag_gemini_tool_query(query, system_prompt, generation_config, safety_settings)
    if hasattr(rag_response, 'text'):
        generated_answer = rag_response.text
    else:
        generated_answer = str(rag_response)
    print("\n=== RAG生成答案 ===\n", generated_answer)

    # 3. 用Gemini 2.5 flash评估
    llm_evaluator = GenerativeModel("gemini-2.5-flash")
    eval_result = evaluate_answer_factuality_with_llm(
        query=query,
        generated_answer=generated_answer,
        retrieved_context=retrieved_context,
        llm_evaluator_model=llm_evaluator
    )
    print("\n=== LLM评估结果 ===\n", eval_result)