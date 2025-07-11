import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from typing import List

# --- 配置您的GCP项目信息 ---
YOUR_PROJECT_ID = "	a94-project-ai-specialization"  # <-- 请替换为您的GCP项目ID
YOUR_REGION = "us-central1"              # <-- 请替换为您的Vertex AI区域

# 初始化Vertex AI SDK
aiplatform.init(project=YOUR_PROJECT_ID, location=YOUR_REGION)

# --- 定义要测试的Embedding模型 ---
EMBEDDING_MODELS_TO_TEST = {
    "Multilingual Embedding (text-multilingual-embedding-002)": "text-multilingual-embedding-002",
    "English/Code Embedding (text-embedding-005)": "text-embedding-005"
}

# --- 定义测试句子 ---
test_sentences = {
    "english_same_1": "The quick brown fox jumps over the lazy dog.",
    "chinese_same_1": "敏捷的棕色狐狸跳过了懒惰的狗。",
    "spanish_same_1": "El veloz zorro marrón salta sobre el perro perezoso.",
    "english_different_2": "Apples are red and grow on trees.",
    "chinese_different_2": "苹果是红色的，长在树上。",
    "english_unrelated": "The capital of France is Paris.",
}

def get_text_embeddings(model_code: str, texts: List[str]) -> List[List[float]]:
    """
    为给定文本列表获取Embedding向量。
    """
    try:
        client = PredictionServiceClient(
            client_options={"api_endpoint": f"{YOUR_REGION}-aiplatform.googleapis.com"}
        )
        endpoint = f"projects/{YOUR_PROJECT_ID}/locations/{YOUR_REGION}/publishers/google/models/{model_code}"
        instances = [{"content": text} for text in texts]
        response = client.predict(endpoint=endpoint, instances=instances)
        embeddings = []
        for prediction in response.predictions:
            # 兼容不同模型的返回结构
            if "embeddings" in prediction and "values" in prediction["embeddings"]:
                embeddings.append(prediction["embeddings"]["values"])
            elif "values" in prediction:
                embeddings.append(prediction["values"])
            else:
                raise ValueError("Unknown embedding response format")
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings for model {model_code}: {e}")
        return []

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return -2.0
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    if np.linalg.norm(vec1_np) == 0 or np.linalg.norm(vec2_np) == 0:
        return 0.0
    return cosine_similarity(vec1_np.reshape(1, -1), vec2_np.reshape(1, -1))[0][0]

# --- 执行测试 ---
print("--- Starting Embedding Model Comparative Test ---")

sentence_keys = list(test_sentences.keys())
sorted_sentences = [test_sentences[key] for key in sentence_keys]

for model_display_name, model_code in EMBEDDING_MODELS_TO_TEST.items():
    print(f"\nEvaluating Model: {model_display_name} ({model_code})")
    all_embeddings = get_text_embeddings(model_code, sorted_sentences)
    if not all_embeddings or len(all_embeddings) != len(sorted_sentences):
        print(f"  Could not get embeddings for all sentences. Skipping this model.")
        continue
    embeddings_map = {key: all_embeddings[i] for i, key in enumerate(sentence_keys)}
    print("  --- Cosine Similarities ---")
    sim1_cross_lang = calculate_cosine_similarity(
        embeddings_map["english_same_1"], embeddings_map["chinese_same_1"]
    )
    print(f"    EN vs ZH (same meaning): {sim1_cross_lang:.4f}")
    sim2_cross_lang = calculate_cosine_similarity(
        embeddings_map["english_same_1"], embeddings_map["spanish_same_1"]
    )
    print(f"    EN vs ES (same meaning): {sim2_cross_lang:.4f}")
    sim3_intra_lang_diff = calculate_cosine_similarity(
        embeddings_map["english_same_1"], embeddings_map["english_different_2"]
    )
    print(f"    EN vs EN (diff meaning): {sim3_intra_lang_diff:.4f}")
    sim4_cross_lang_diff = calculate_cosine_similarity(
        embeddings_map["english_same_1"], embeddings_map["chinese_different_2"]
    )
    print(f"    EN vs ZH (diff meaning): {sim4_cross_lang_diff:.4f}")
    sim5_en_baseline = calculate_cosine_similarity(
        embeddings_map["english_same_1"], embeddings_map["english_same_1"]
    )
    print(f"    EN vs EN (identical): {sim5_en_baseline:.4f}")

print("\n--- Test Finished ---")