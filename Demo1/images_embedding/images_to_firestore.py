import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from google.cloud import storage
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
import time

PROJECT_ID = "a94-project-ai-specialization"
vertexai.init(project=PROJECT_ID, location="us-central1")

# GCS配置
BUCKET_NAME = "ragdiydemo"
PREFIX = "Image_source_data/"
PUBLIC_URL_PREFIX = "https://storage.googleapis.com/ragdiydemo/Image_source_data/"

# Firestore配置
firestore_client = firestore.Client()
collection = firestore_client.collection("images-demo")

# 初始化模型
embedding_dimension = 512
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

def get_image_embedding(image_gcs_path: str, contextual_text: str = "") -> list:
    """
    获取图片的embedding向量
    """
    try:
        image = Image.load_from_file(image_gcs_path)
        embeddings = model.get_embeddings(
            image=image,
            contextual_text=contextual_text,
            dimension=embedding_dimension,
        )
        return list(embeddings.image_embedding)
    except Exception as e:
        print(f"获取embedding失败 {image_gcs_path}: {e}")
        return None

def process_images():
    """
    遍历GCS中的所有图片，生成embeddings并存储到Firestore
    """
    # 初始化GCS客户端
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # 获取所有文件夹
    blobs = bucket.list_blobs(prefix=PREFIX)
    folders = set()
    
    # 提取文件夹名称
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            # 从路径中提取文件夹名称
            # 例如: Image_source_data/cat/001fc748e6.jpg -> cat
            path_parts = blob.name.split('/')
            if len(path_parts) >= 3:
                folder_name = path_parts[1]  # cat
                folders.add(folder_name)
    
    print(f"发现文件夹: {sorted(folders)}")
    
    # 为每个图片分配唯一ID
    image_id = 1
    
    # 处理每个文件夹
    for folder in sorted(folders):
        print(f"\n处理文件夹: {folder}")
        
        # 获取该文件夹下的所有jpg文件
        folder_prefix = f"{PREFIX}{folder}/"
        folder_blobs = bucket.list_blobs(prefix=folder_prefix)
        
        for blob in folder_blobs:
            if not blob.name.endswith('.jpg'):
                continue
                
            # 构建GCS路径和公开URL
            gcs_path = f"gs://{BUCKET_NAME}/{blob.name}"
            file_name = blob.name.split('/')[-1]  # 001fc748e6.jpg
            public_url = f"{PUBLIC_URL_PREFIX}{folder}/{file_name}"
            
            print(f"处理图片 {image_id}: {public_url}")
            
            # 获取embedding，使用folder_name作为contextual_text
            embedding = get_image_embedding(gcs_path, contextual_text=folder)
            
            if embedding:
                # 构建Firestore文档
                doc = {
                    "id": str(image_id),
                    "embedding_field": Vector(embedding),
                    "path": public_url
                }
                
                # 存储到Firestore
                try:
                    collection.add(doc)
                    print(f"成功存储图片 {image_id} 到Firestore")
                except Exception as e:
                    print(f"存储到Firestore失败 {image_id}: {e}")
                
                image_id += 1
                
                # 添加延迟避免API限制
                time.sleep(0.1)
            else:
                print(f"跳过图片 {image_id} (embedding获取失败)")
                image_id += 1
    
    print(f"\n处理完成！总共处理了 {image_id - 1} 张图片")

if __name__ == "__main__":
    print("开始处理图片embeddings...")
    process_images() 