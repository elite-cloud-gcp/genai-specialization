# Cloud Function 部署说明

## 功能描述
这个 Cloud Function 会在接收到 Pub/Sub 消息时，自动将指定的文件导入到 RAG corpus 中。

## 部署步骤

### 1. 配置环境变量
在部署 Cloud Function 时，需要设置以下环境变量：

```bash
PROJECT_ID=a94-project-ai-specialization
LOCATION=us-central1
CORPUS_NAME=your_actual_corpus_name
```

### 2. 获取实际的 Corpus 名称
运行以下命令获取已创建的 RAG corpus 名称：

```bash
gcloud ai rag corpora list --region=us-central1 --project=a94-project-ai-specialization
```

然后将实际的 corpus 名称替换 `cloud_function.py` 中的 `CORPUS_NAME` 变量。

### 3. 部署 Cloud Function

```bash
gcloud functions deploy rag-file-import \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=hello_pubsub \
  --trigger-topic=your-pubsub-topic-name \
  --set-env-vars=PROJECT_ID=a94-project-ai-specialization,LOCATION=us-central1
```

### 4. 设置 Pub/Sub 触发器
确保 Pub/Sub 消息格式包含以下字段：
```json
{
  "name": "filename.txt",
  "bucket": "your-bucket-name"
}
```

### 5. 权限配置
确保 Cloud Function 的服务账户具有以下权限：
- Vertex AI User
- Storage Object Viewer
- Pub/Sub Subscriber

## 测试
可以通过发送测试消息到 Pub/Sub 主题来测试功能：

```bash
gcloud pubsub topics publish your-topic-name \
  --message='{"name":"test-file.txt","bucket":"your-bucket-name"}'
```

## 注意事项
1. 确保文件路径格式为 `gs://bucket-name/filename`
2. 文件必须存在于指定的 GCS bucket 中
3. Corpus 必须已经创建并配置好
4. 检查 Cloud Function 日志以监控导入状态 