from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

# 初始化 Vertex AI
PROJECT_ID = "a94-project-ai-specialization"
vertexai.init(project=PROJECT_ID, location="us-central1")

# 1. 生成 demo.jpg 的 embedding
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
image = Image.load_from_file(
    "gs://ragdiydemo/Image_source_data/demo.jpg"
)
embeddings = model.get_embeddings(
    image=image,
    contextual_text="",  # 可根据需要填写
    dimension=512,
)
embedding_vector = list(embeddings.image_embedding)

# 2. 用 embedding_vector 查询 Firestore
firestore_client = firestore.Client()
collection = firestore_client.collection("images-demo")

vector_query = collection.find_nearest(
    vector_field="embedding_field",
    query_vector=Vector(embedding_vector),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=5,
)


# 3. 根据每个doc的"id"字段，再查一次Firestore，获取path
print("匹配到的5条结果及其图片URL：")
for doc in vector_query.stream():
    doc_id = doc.get("id")
    # 查询id字段等于doc_id的文档
    docs = collection.where("id", "==", doc_id).limit(1).stream()
    for d in docs:
        data = d.to_dict()
        print(f"id: {doc_id}, path: {data.get('path')}")



