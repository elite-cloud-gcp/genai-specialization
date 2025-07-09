import gradio as gr
import tempfile
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

# Vertex AI & Firestore 配置
PROJECT_ID = "a94-project-ai-specialization"
vertexai.init(project=PROJECT_ID, location="us-central1")
firestore_client = firestore.Client()
collection = firestore_client.collection("images-demo")

# 初始化模型
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# -------- 文本RAG检索相关 --------
corpus_name = "projects/a94-project-ai-specialization/locations/us-central1/ragCorpora/3379951520341557248"

def rag_gemini_tool_query(query_text: str):
    try:
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
        return response.text
    except Exception as e:
        return f"查询出错: {str(e)}"

def query_rag(query_text):
    if not query_text.strip():
        return "请输入查询内容"
    result = rag_gemini_tool_query(query_text)
    return result

# -------- 图片向量检索相关 --------
def get_image_embedding_local(image_path, dimension=512):
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(
        image=image,
        contextual_text="",  # 可根据需要填写
        dimension=dimension,
    )
    return list(embeddings.image_embedding)

def find_similar_images(embedding_vector, top_k=5):
    vector_query = collection.find_nearest(
        vector_field="embedding_field",
        query_vector=Vector(embedding_vector),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=top_k,
    )
    urls = []
    for doc in vector_query.stream():
        doc_id = doc.get("id")
        docs = collection.where("id", "==", doc_id).limit(1).stream()
        for d in docs:
            data = d.to_dict()
            if "path" in data:
                urls.append(data["path"])
    return urls

def gradio_image_search(input_image, top_k):
    # 保存上传图片到临时文件
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        input_image.save(tmp.name)
        image_path = tmp.name

    # 生成embedding
    embedding_vector = get_image_embedding_local(image_path)

    # 查询最近邻图片
    urls = find_similar_images(embedding_vector, top_k=top_k)

    # 返回图片URL列表，gradio会自动展示
    return urls

# -------- Gradio多标签页UI --------
with gr.Blocks(title="RAG+图片向量检索") as demo:
    gr.Markdown("# 🔎 智能检索系统\n支持文本RAG智能问答和图片相似检索。")

    with gr.Tab("文本RAG智能检索"):
        gr.Markdown("### 🤖 文本RAG智能检索\n请输入您的问题，系统将基于RAG技术为您提供智能回答。")
        query_input = gr.Textbox(
            label="请输入您的查询",
            placeholder="例如：我想购买Vishudh品牌的产品，有什么推荐且价格便宜的吗？",
            lines=4,
            max_lines=8,
            container=True,
            scale=1
        )
        query_button = gr.Button("🔍 开始查询", variant="primary", size="lg", scale=1)
        result_output = gr.Textbox(
            label="查询结果",
            lines=10,
            max_lines=20,
            interactive=False,
            container=True,
            scale=1
        )
        query_button.click(
            fn=query_rag,
            inputs=query_input,
            outputs=result_output
        )
        query_input.submit(
            fn=query_rag,
            inputs=query_input,
            outputs=result_output
        )

    with gr.Tab("图片向量检索"):
        gr.Markdown("### 🖼️ 图片向量检索\n上传一张图片，检索图库中最相似的图片。")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上传图片")
                topk_slider = gr.Slider(1, 10, value=5, step=1, label="TopK 最邻近数量")
                search_btn = gr.Button("开始检索")
            with gr.Column():
                gallery = gr.Gallery(label="相似图片展示", show_label=True, columns=5, height="auto")

        search_btn.click(
            fn=gradio_image_search,
            inputs=[input_image, topk_slider],
            outputs=gallery
        )
        input_image.change(
            fn=gradio_image_search,
            inputs=[input_image, topk_slider],
            outputs=gallery
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 