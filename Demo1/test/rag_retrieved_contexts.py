import sys
import json
sys.path.append("../rag_engine")
from rag_query import retrieval_query, rag_gemini_tool_query

questions = [
"What is the unique advantage of Google Cloud Platform (GCP) in the cloud computing market?",
"How does Google stand out in the current 'cloud wars'?",
"Why is Google considered one of the winners of the 'cloud wars'?",
"Besides price, what are the key factors that enterprises need to consider when choosing a cloud platform?",
"Why do enterprises need a 'flexible cloud strategy'?",
"What are the main business drivers of cloud computing for startups and small and medium-sized businesses (SMBs)?",
"How does cloud help enterprises convert capital expenditures into operating expenses?",
"How does cloud computing help enterprises in improving enterprise scalability?",
"How does cloud facilitate enterprise mobility and BYOD policies?",
"How does cloud computing accelerate new product development and time to market?",
"How should we evaluate the performance and uptime of cloud service providers?",
"How important are service level agreements (SLAs) and reliability when choosing a cloud provider?",
"How to compare the cost models of different cloud providers (e.g., hourly billing vs. minute-by-minute billing)?",
"What are the advantages of choosing a cloud service provider that supports your existing technology stack?",
"How to avoid or mitigate vendor lock-in in cloud platforms?",
"What guarantees should cloud service providers provide in terms of data security and compliance?",
"Does Google Cloud Platform support hybrid cloud deployment models?",
"Why should you care about the regional support capabilities of cloud providers?",
"How does autoscaling help manage application load fluctuations?",
"How important is network connectivity for running latency-sensitive cloud applications?",
"Why did Spotify choose to migrate to Google Cloud Platform (GCP)?",
"What success did Snapchat achieve with Google App Engine?",
"What lessons can we learn from Apple's strategy in choosing cloud platforms?",
"How should we formulate cloud computing strategies as a startup?",
"What is the recommended roadmap for formulating an enterprise cloud strategy?",
"How to measure and evaluate the success of enterprise cloud migration?",
"What are the fundamental building blocks (Fundamental Services) of cloud platforms?",
"What does 'Compute' mean in cloud computing?",
"What types of cloud storage services are usually included?",
"Why are cloud network services fundamental and important?",
"What are the categories and characteristics of cloud database services?",
"How are higher-level services built on top of base services?",
"What are the core architectural similarities between Google Cloud and AWS?",
"What are the 'Application Services' provided by Google?",
"What are the 'Data Services' provided by Google?",
"What are the 'Management Services' provided by Google?",
"What are the key differences between Google Compute Engine (GCE) and Amazon EC2?",
"What virtual machine (VM) instance types does GCE support?",
"What are the differences between Shared Core VMs and Standard VMs and when to choose one over the other?",
"Does GCP offer GPU-enabled VMs?",
"How are local SSDs used in GCE and what are the performance benefits?",
"Does GCE support dense storage VMs?",
"What are the OS support and licensing options for GCE?",
"How do I configure firewall rules in Google Compute Engine?",
"How does Google's autoscaling feature compare to Amazon's ",
"How do I import an existing VM image into Google Cloud Platform?",
"How does Google Compute Engine's billing model (e.g., per-minute billing) compare to competitors? ",
"How does Google's Sustained Use Discount work and how much can I save?",
"How do Amazon Elastic Beanstalk and Google App Engine (GAE) differ in terms of managed services? ",
"Is GAE a good choice for projects that need a quick start and fully managed service?",
"What are the main differences between Google Cloud Storage (GCS) and Amazon S3? ",
"What scenarios are suitable for the 'hot', 'warm', 'cold', and 'ultra-cold' data storage categories of Cloud Storage? ",
"What are the main differences between Google Cloud Storage Nearline and Amazon Glacier? ",
"How does Nearline compare to Glacier in terms of data retrieval speed and cost?",
"Are there any data retrieval rate limits for Google Nearline?",
"How does Nearline compare to Glacier in terms of data overwrite and deletion policies? ",
"What are the differences? Which one is more suitable for frequent deletion scenarios? ",
"Does GCS support chunked encoding and resumable uploads? ",
"What is the coverage of Google Cloud Storage in global data center regions? ",
"Can GCS be seamlessly integrated with other Google services like S3? ",
"When choosing a cloud storage service, what other factors should be considered besides capacity price? ",
"What are the 'durability' and 'availability' of cloud storage? ",
"What is the use of the 'lifecycle management' feature of cloud storage? ",
"How does Google attract non-GCP users with its storage services? ",
"Why is network reliability key to cloud applications? ",
"How is Google Cloud's network architecture design different from Amazon's? ",
"What are the characteristics of Google Andromeda's network architecture? ",
"What is the difference between Google Cloud Load Balancer and Amazon ELB in terms of real-time responsiveness? ",
"What is the pricing model of Google Cloud Load Balancer? ",
"What is peering in cloud networking and what does it do?",
"How does Google Cloud's VPN service keep your data secure?",
"Does Google Cloud offer direct peering?",
"How does CDN Interconnect work in Google Cloud?",
"How does pricing for the Google Cloud and AWS interconnect services differ?",
"Is Google Cloud DNS functionally equivalent to Amazon Route 53?",
"What are the benefits of Google Cloud's Live Migration feature?",
"How does Google Cloud perform in terms of network performance, according to independent expert benchmarks?",
"How can database as a service (DBaaS) help enterprises reduce database sprawl?",
"How does DBaaS enable automatic scaling and high availability of databases?",
"What features does DBaaS provide to keep databases secure?",
"What database management tasks (e.g., patching, backups) are typically handled by DBaaS providers?",
"Google Cloud SQL vs. Amazon RDS  What is the comparison between functions and performance? ", 
"What are the characteristics of Google Cloud SQL in terms of scalability, connectivity, and startup flexibility?", 
"How is the price-performance ratio of Google Cloud SQL?", 
"What type of database is Google Cloud Datastore and what scenarios is it suitable for?", 
"What are the main use cases for Google Cloud Bigtable? How is it different from Amazon DynamoDB?", 
"What are the differences in pricing strategies between Google Bigtable and Amazon DynamoDB when used on demand?", 
"Does Google Bigtable differentiate pricing for read and write operations?", 
"How does cloud computing achieve the 'democratization' of big data analysis?", 
"How does Google Cloud's big data analytics service help companies like Spotify and Dominos?", 
"What are the characteristics of Google Cloud Dataflow compared to Amazon EMR/Data Pipeline?", 
"What has been the pricing of Google BigQuery recently adjusted?", 
"What features does Google Data Studio 360 provide in data visualization?", 
"How does Google BigQuery's 'No Ops' model simplify big data processing?", 
"Why does Spotify think Google ",
"How does the Internet of Things (IoT) drive further adoption of Google Cloud Platform big data services? ",
"How does machine learning (ML) bring predictive and prescriptive insights to enterprises? ",
"What is the functional difference between Amazon ML and Google Machine Learning Platform? ",
"What are the two main components of Google Machine Learning Platform? ",
"What pre-trained machine learning APIs (e.g., vision, translation, speech) does Google Cloud offer? ",
"Why is Google considered the clear leader in machine learning as a service (MLaaS)? "
]


# 生成RAGretrieved_contexts
retrieved_contexts = []

for q in questions:
    retrieval_response = retrieval_query(q)
    # 兼容 RagContexts 对象
    if hasattr(retrieval_response, 'contexts'):
        contexts_list = retrieval_response.contexts.contexts
        if contexts_list and hasattr(contexts_list[0], 'text'):
            retrieved_contexts.append(contexts_list[0].text)
        else:
            retrieved_contexts.append("")
    elif isinstance(retrieval_response, dict) and 'contexts' in retrieval_response:
        if retrieval_response['contexts'] and 'text' in retrieval_response['contexts'][0]:
            retrieved_contexts.append(retrieval_response['contexts'][0]['text'])
        else:
            retrieved_contexts.append("")
    else:
        retrieved_contexts.append("")

# 保存为JSON文件
with open("retrieved_contexts.json", "w", encoding="utf-8") as f:
    json.dump(retrieved_contexts, f, ensure_ascii=False, indent=2)

# 生成RAG答案
system_prompt = "You are a helpful assistant who can answer questions about the products in the corpus and other assorted questions."
generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.9,
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

generated_answers_by_rag = []
for q in questions:
    rag_response = rag_gemini_tool_query(q, system_prompt, generation_config, safety_settings)
    if hasattr(rag_response, 'text'):
        generated_answers_by_rag.append(rag_response.text)
    else:
        generated_answers_by_rag.append(str(rag_response))

# 保存为JSON文件
with open("generated_answers_by_rag.json", "w", encoding="utf-8") as f:
    json.dump(generated_answers_by_rag, f, ensure_ascii=False, indent=2)

# 输出统计信息
print(f"总共有 {len(questions)} 条问题")
print(f"总共有 {len(retrieved_contexts)} 条检索上下文")
print(f"总共有 {len(generated_answers_by_rag)} 条RAG生成答案")
print("数据已保存到 retrieved_contexts.json 和 generated_answers_by_rag.json")