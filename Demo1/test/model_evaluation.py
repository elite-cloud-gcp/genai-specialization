import pandas as pd
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples, PointwiseMetric
from vertexai.preview.evaluation import notebook_utils
import os
from google.cloud import storage
import tempfile
import json

PROJECT_ID = "a94-project-ai-specialization"  
LOCATION = "us-central1"  
EXPERIMENT = "rag-eval-01" 

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)

# GCS文件路径
questions_gcs = "gs://ragdiydemo/model_evaluation/questions.json"
contexts_gcs = "gs://ragdiydemo/model_evaluation/retrieved_contexts.json"
answers_gcs = "gs://ragdiydemo/model_evaluation/generated_answers_by_rag.json"

def download_gcs_file(gcs_path, local_path):
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS路径必须以gs://开头")
    bucket_name, blob_path = gcs_path[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

# 下载并导入JSON文件
with tempfile.TemporaryDirectory() as tmpdir:
    local_questions = os.path.join(tmpdir, "questions.json")
    local_contexts = os.path.join(tmpdir, "retrieved_contexts.json")
    local_answers = os.path.join(tmpdir, "generated_answers_by_rag.json")
    download_gcs_file(questions_gcs, local_questions)
    download_gcs_file(contexts_gcs, local_contexts)
    download_gcs_file(answers_gcs, local_answers)

    # 直接使用json.load解析JSON文件
    with open(local_questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(local_contexts, "r", encoding="utf-8") as f:
        retrieved_contexts = json.load(f)
    with open(local_answers, "r", encoding="utf-8") as f:
        generated_answers_by_rag = json.load(f)

# 变量已在全局命名空间中
print(f"questions: {len(questions)} 条")
print(f"retrieved_contexts: {len(retrieved_contexts)} 条")
print(f"generated_answers_by_rag: {len(generated_answers_by_rag)} 条")

# 示例：打印前3条
for i in range(min(3, len(questions))):
    print(f"Q{i+1}: {questions[i]}\nContext: {retrieved_contexts[i]}\nAnswer: {generated_answers_by_rag[i]}\n{'-'*40}")

eval_dataset_rag = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + item
            for question, item in zip(questions, retrieved_contexts)
        ],
        "response": generated_answers_by_rag,
    }
)

relevance_prompt_template = """
You are a professional cloud computing assessor. Your job is to score the responses according to the pre-set assessment criteria.

You will assess relevance, which is the ability to respond with relevant information when prompted.

You will score the written response 5, 4, 3, 2, 1 according to the rubric and assessment steps.

## Criteria
Relevance: The response should be relevant to the cloud computing content and directly address cloud computing.

## Scoring Criteria
5 (Completely relevant): The response is completely relevant to cloud computing and provides clear and unambiguous information that directly addresses the core needs of cloud computing.

4 (Partially relevant): The response is mostly relevant to cloud computing and mostly directly addresses cloud computing.

3 (Partially relevant): The response is partially relevant to cloud computing and may indirectly address cloud computing, but can be more relevant and more directly.

2 (Partially irrelevant): The response is mostly relevant to cloud computing and does not directly address cloud computing.

1 (Irrelevant): The answer is completely irrelevant to cloud computing.

## Assessment Steps
Step 1: Assess relevance: Is the answer relevant to cloud computing and directly address cloud computing?

Step 2: Score according to the criteria and rubric.

Please explain your scoring steps step by step and choose scores only from 5, 4, 3, 2, 1.

# User input and AI generated answers
## User input
### Prompt
{prompt}

## AI generated answers
{response}
"""

relevance = PointwiseMetric(
    metric="relevance",
    metric_prompt_template=relevance_prompt_template,
)

rag_eval_task_rag = EvalTask(
    dataset=eval_dataset_rag,
    metrics=[
        "question_answering_quality",
        relevance,
        "groundedness",
        "safety",
        "instruction_following",
    ],
    experiment=EXPERIMENT,
)

result_rag = rag_eval_task_rag.evaluate()

# 调试：检查结果对象
print("=== 调试结果对象 ===")
print(f"结果对象类型: {type(result_rag)}")
print(f"结果对象属性: {dir(result_rag)}")
print(f"结果对象内容: {result_rag}")

# 尝试不同的属性访问方式
if hasattr(result_rag, 'metrics'):
    print(f"metrics 属性存在: {result_rag.metrics}")
else:
    print("metrics 属性不存在")

if hasattr(result_rag, 'results'):
    print(f"results 属性存在: {result_rag.results}")
else:
    print("results 属性不存在")

# 打印整个结果对象的字符串表示
print(f"完整结果: {str(result_rag)}")

# 打印评估结果
print("=== 评估结果 ===")
print(f"评估完成，共处理 {len(eval_dataset_rag)} 条数据")

# 保存结果到文件
import json
import numpy as np

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    results_dict = {
        "experiment": EXPERIMENT,
        "dataset_size": len(eval_dataset_rag),
        "summary_metrics": convert_numpy_types(result_rag.summary_metrics),
        "metadata": result_rag.metadata
    }
    
    # 保存详细的metrics_table（前几行作为示例）
    if hasattr(result_rag, 'metrics_table') and result_rag.metrics_table is not None:
        # 只保存前5行作为示例，避免文件过大
        sample_data = result_rag.metrics_table.head(5).to_dict('records')
        results_dict["metrics_table_sample"] = convert_numpy_types(sample_data)
    
    json.dump(results_dict, f, ensure_ascii=False, indent=2)

print("评估结果已保存到 evaluation_results.json")
print("=== 评估总结 ===")
for key, value in result_rag.summary_metrics.items():
    if 'mean' in key:
        print(f"{key}: {value:.3f}")