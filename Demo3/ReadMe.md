### EliteCloud Demo for Supervised Fine-Tuning of Gemini 2.0 Flash for Article Summarization. 
This demo provides a step-by-step guide for fine-tuning the gemini-2.0-flash-001 model for article summarization using Vertex AI. The process covers everything from data preparation to model evaluation and deployment.

### 🚀 Overview
Running a supervised tuning job on Vertex AI.Evaluating the performance of both the base and the fine-tuned models.Using the new model for inference.

### ✅ Prerequisites
Before you start, make sure you have:A Google Cloud Project.The Vertex AI API enabled.A Google Cloud Storage (GCS) bucket to store your datasets.Authenticated your local or Colab environment to access Google Cloud.

### 💻 Installation
Install the necessary Python libraries using pip:pip install --upgrade --user google-genai google-cloud-aiplatform rouge_score plotly jsonlines
Note: If you are in a Google Colab environment, you may need to restart the runtime after installation.

### 📚 Dataset
This project utilizes the Wikilingua dataset, which contains article-summary pairs from WikiHow across multiple languages. We focus on the English portion of the dataset for this tutorial.Data FormatThe training data must be in JSONL format, where each line is a JSON object. Each object must contain a contents array that alternates between user (the input article) and model (the target summary) roles.
Example JSONL line:
```python
{
   "contents":[
      {
         "role":"user",
         "parts":[
            {"text": "Full text of the article goes here..."}
         ]
      },
      {
         "role":"model",
         "parts":[
            {"text": "The expected summary of the article goes here."}
         ]
      }
   ]
}
Citation@inproceedings{ladhak-wiki-2020,
    title={WikiLingua: A New Benchmark Dataset for Multilingual Abstractive Summarization},
    author={Faisal Ladhak, Esin Durmus, Claire Cardie and Kathleen McKeown},
    booktitle={Findings of EMNLP, 2020},
    year={2020}
}
```

### 🛠️ Usage
Follow these steps to fine-tune the model.1. Configure EnvironmentSet up your Project ID and Region, then initialize the Vertex AI SDK.
```python
import vertexai
import google.generativeai as genai

PROJECT_ID = "your-gcp-project-id"  # <--- Change this
REGION = "us-central1"        # <--- Change this

vertexai.init(project=PROJECT_ID, location=REGION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
```
2. Prepare & Upload DataCreate a GCS bucket and upload your formatted training and validation files.
```python
BUCKET_NAME = "your-bucket-name" # <--- Change this
BUCKET_URI = f"gs://{BUCKET_NAME}"
```
```
### Create the bucket (if it doesn't exist)
!gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}
```
### Copy your local data to the bucket
```
!gsutil cp sft_train_samples.jsonl {BUCKET_URI}/train/
!gsutil cp sft_val_samples.jsonl {BUCKET_URI}/val/
```
3. Evaluate the Base ModelBefore tuning, it's a good practice to evaluate the performance of the original gemini-2.0-flash-001 model on a test set. This provides a baseline to measure improvement against.4. Run the Fine-Tuning JobStart the supervised tuning job on Vertex AI. This process can take some time to complete.

```python
from google.genai import types

tuned_model_display_name = "gemini-flash-summarization-v1" # <--- Change this

sft_tuning_job = client.tunings.tune(
    base_model="gemini-2.0-flash-001",
    training_dataset={
        "gcs_uri": f"{BUCKET_URI}/train/sft_train_samples.jsonl",
    },
    config=types.CreateTuningJobConfig(
        tuned_model_display_name=tuned_model_display_name,
        validation_dataset=types.TuningValidationDataset(
            gcs_uri=f"{BUCKET_URI}/val/sft_val_samples.jsonl"
        ),
        # Optional hyperparameters:
        # epochs=4,
        # learning_rate_multiplier=1.0,
    ),
)
```


#### You can check the status of the job
```python
print(sft_tuning_job.state)
```

5. Evaluate the Tuned ModelOnce the job is complete, a new model endpoint will be created. Evaluate this new model on the same test set to quantify the performance improvement.

### 📊 Evaluation
The primary metric used is ROUGE-L, which evaluates summary quality by measuring the longest common subsequence between the generated summary and a human-written reference summary. An increase in the ROUGE-L score indicates an improvement in summarization capability.The notebook provides code to plot the training and evaluation loss curves, which helps visualize the learning process.

### 🧹 Cleaning Up
To avoid incurring future costs, delete the cloud resources created during this tutorial.
#### Delete the Vertex AI Endpoint
```python
# endpoint.delete(force=True)
# Delete the Cloud Storage Bucket and its contents
# !gsutil -m rm -r $BUCKET_URI
```
### 📄 License
This project is licensed under the Apache License, Version 2.0. See the LICENSE file for more details.