# DIY RAG Demo 

这个项目实现了一个完整的 RAG (Retrieval-Augmented Generation) 系统，包括文件导入、向量搜索和查询功能。

## 项目结构

```
diy-rag-demo/
├── cloud_function.py          # Cloud Function 主文件
├── requirements.txt           # Python 依赖
├── DEPLOYMENT.md             # 部署说明
├── test_cloud_function.py    # 测试脚本
├── rag_engine/               # RAG 引擎模块
│   ├── create_ragcorpus.py   # 创建 RAG 语料库
│   ├── create_vectorsearch.py # 创建向量搜索
│   ├── data_cleaing.py       # 数据清理
│   └── rag_query.py          # RAG 查询
├── images_embedding/         # 图像嵌入模块
│   ├── images_query.py       # 图像查询
│   └── images_to_firestore.py # 图像到 Firestore
└── gradio_ui_all.py         # Gradio UI 界面
```

## 主要功能

### 1. Cloud Function 自动文件导入
- 监听 Pub/Sub 消息
- 自动将新上传的文件导入到 RAG corpus
- 支持文本分块和向量化

### 2. RAG 语料库管理
- 创建和管理 RAG corpus
- 配置向量搜索索引
- 文件导入和分块处理

### 3. 查询功能
- 文本查询和检索
- 图像查询和检索
- 支持多种查询模式

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export PROJECT_ID="your-project-id"
export LOCATION="us-central1"
export CORPUS_NAME="your-corpus-name"
```

### 2. 创建 RAG Corpus
```bash
python rag_engine/create_ragcorpus.py
```

### 3. 部署 Cloud Function
```bash
# 参考 DEPLOYMENT.md 中的详细步骤
gcloud functions deploy rag-file-import \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=hello_pubsub \
  --trigger-topic=your-pubsub-topic-name
```

### 4. 测试功能
```bash
# 运行测试脚本
python test_cloud_function.py

# 或发送测试消息到 Pub/Sub
gcloud pubsub topics publish your-topic-name \
  --message='{"name":"test-file.txt","bucket":"your-bucket-name"}'
```

## 配置说明

### 环境变量
- `PROJECT_ID`: Google Cloud 项目 ID
- `LOCATION`: Vertex AI 区域
- `CORPUS_NAME`: RAG corpus 名称
- `CHUNK_SIZE`: 文本分块大小（默认 512）
- `CHUNK_OVERLAP`: 分块重叠大小（默认 100）

### Pub/Sub 消息格式
```json
{
  "name": "filename.txt",
  "bucket": "your-bucket-name"
}
```

## 工作流程

1. **文件上传**: 用户上传文件到 GCS bucket
2. **触发通知**: GCS 发送 Pub/Sub 消息
3. **自动处理**: Cloud Function 接收消息并导入文件到 RAG corpus
4. **向量化**: 文件被分块并转换为向量
5. **查询**: 用户可以通过 UI 或 API 查询内容

## 监控和日志

- Cloud Function 日志: Google Cloud Console > Functions
- RAG corpus 状态: Vertex AI > RAG
- 向量搜索: Vertex AI > Vector Search

## 故障排除

### 常见问题

1. **权限错误**: 确保服务账户有正确的 IAM 权限
2. **Corpus 不存在**: 先运行 `create_ragcorpus.py` 创建 corpus
3. **文件路径错误**: 确保文件存在于指定的 GCS bucket 中
4. **依赖问题**: 检查 `requirements.txt` 中的依赖版本

### 调试步骤

1. 检查 Cloud Function 日志
2. 验证环境变量设置
3. 确认 corpus 名称正确
4. 测试文件访问权限

## 扩展功能

- 支持更多文件格式
- 添加文件预处理
- 实现批量导入
- 集成更多 AI 模型
