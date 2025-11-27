# Demo 4 

This project demonstrates a complete machine learning workflow for predicting taxi fares in Chicago. It covers data exploration, feature engineering, model training, and deployment on Google Cloud's Vertex AI, including the use of Explainable AI.

## Project Structure
Demo4/

├── config.py                   # Global configuration
├── 1_data_exploration.ipynb    # ML 3.4.3.2: EDA
├── src/

│   ├── preprocessing.sql       # ML 3.4.3.4: BigQuery Logic

│   ├── train_model.py          # ML 3.4.3.3 - 3.4.3.7: Feature Eng, Training, Eval

│   └── pipeline_job.py         # ML 3.4.3.6: Vertex AI Training Job submission

└── 2_deploy_and_predict.py     # ML 3.4.4: Deployment & Prediction Proof


-----

## 1\. Business Goal and Machine Learning Solution

### The Business Question/Goal Being Addressed

The primary business goal is to **accurately predict the total fare (`trip_total`) for a taxi trip in Chicago**. This predictive capability can be valuable for various stakeholders, including customers who want fare estimates, taxi companies for optimizing pricing strategies, and drivers for understanding potential earnings.

### The Machine Learning Use Case

This project addresses a classic **regression** use case. The objective is to build a model that can predict a continuous numerical value—the taxi fare—based on various features related to the trip.

### How the Machine Learning Solution is Expected to Address the Business Question/Goal

The machine learning solution directly addresses the business goal by providing a **Linear Regression model** that takes trip characteristics as input and outputs a predicted fare. This model provides a data-driven and automated way to estimate fares, replacing simple heuristics or manual calculations. By deploying this model on Vertex AI, it becomes a scalable and accessible tool that can be integrated into applications to provide real-time fare predictions.

-----

## 2\. Data Exploration

### How and What Type of Data Exploration Was Performed

Data exploration was performed to understand the structure, quality, and statistical properties of the Chicago Taxi Trips dataset. The following methods were used:

  * **Initial Data Loading and Inspection**: A subset of the data from May 2018 was loaded from BigQuery. Initial checks included examining the data's shape, column types (`dtypes`), and looking for null values (`df.info()`).
  * **Numerical Data Analysis**:
      * **Descriptive Statistics**: The `.describe()` method was used to get a summary of the central tendency, dispersion, and shape of the numerical columns' distribution.
      * **Visualizations**: Histograms and box plots were generated for each numerical feature (`trip_seconds`, `trip_miles`, `trip_total`) to visualize their distributions and identify outliers. A pair-plot was also created on a sample of the data to understand the bivariate relationships between numerical variables.
  * **Categorical Data Analysis**:
      * **Value Counts**: The distribution of levels within categorical features (`payment_type`, `pickup_community_area`, `dropoff_community_area`) was analyzed using `value_counts()`.
      * **Box Plots**: The relationship between categorical features and the target variable (`trip_total`) was visualized using box plots to see how fares varied across different categories.

### What Decisions Were Influenced by Data Exploration

The insights gained from data exploration directly influenced subsequent preprocessing and modeling decisions:

  * **Outlier Removal**: The histograms and box plots revealed significant outliers (e.g., trips with extremely high speed or duration). This led to the decision to filter the dataset based on logical constraints (e.g., `trip_total > 3`, `trip_hours <= 2`, `trip_speed <= 70 mph`) to remove improbable or erroneous data points. A specific outlier with a fare over $3000 was also removed.
  * **Feature Selection**: The pair-plot showed a strong linear relationship between `trip_seconds`, `trip_miles`, and the target `trip_total`, confirming their importance as predictive features.
  * **Feature Engineering**: Observing the `trip_start_timestamp` field led to the creation of `dayofweek` and `hour` features to capture potential temporal patterns in fare pricing.
  * **Data Cleaning**: The analysis of `payment_type` showed that most trips were paid by `Credit Card` or `Cash`. This justified the decision to filter out other less frequent payment methods to simplify the model.

-----

## 3\. Feature Engineering

### What Feature Engineering Was Performed

The following feature engineering steps were performed to create more meaningful variables for the model:

  * **Time Conversion**: The `trip_seconds` feature was converted into `trip_hours` to represent the trip duration in a more interpretable unit.
  * **Speed Calculation**: A `trip_speed` feature was created by dividing `trip_miles` by `trip_hours`. This feature captures the efficiency of the trip and can be an important predictor of the fare.
  * **Timestamp Decomposition**: The `trip_start_timestamp` was decomposed to extract temporal features:
      * `dayofweek`: The day of the week the trip started.
      * `hour`: The hour of the day the trip started.
  * **Feature Binning and Encoding**: The newly created temporal features were bucketed to reduce complexity and capture broader patterns:
      * `dayofweek` was converted into a binary feature representing **weekday (1)** vs. **weekend (0)**.
      * `hour` was converted into a binary feature representing **working hours (1)** vs. **non-working hours (0)**.

### What Features Were Selected for Use in the Machine Learning Model and Why

The final set of features selected for training the machine learning model was:

  * `trip_seconds`: The duration of the trip in seconds.
  * `trip_miles`: The distance of the trip in miles.
  * `payment_type`: Encoded payment method (Credit Card/Cash).
  * `pickup_community_area`: The community area where the trip started.
  * `dropoff_community_area`: The community area where the trip ended.
  * `dayofweek`: Binned and encoded feature for weekday vs. weekend.
  * `hour`: Binned and encoded feature for working vs. non-working hours.
  * `trip_speed`: The calculated speed of the trip in miles per hour.

These features were chosen because the data exploration and feature engineering steps indicated they had a logical and statistically significant relationship with the target variable, `trip_total`. They represent a mix of duration, distance, location, time, and trip efficiency, which are all intuitive drivers of taxi fare.

-----

## 4\. Preprocessing and the Data Pipeline

The data preprocessing pipeline consists of a series of sequential steps to clean, transform, and prepare the raw data for modeling. While not encapsulated in a single callable function in the notebook, these steps form a logical pipeline:

1.  **Data Ingestion**: Data is initially queried and loaded from the `bigquery-public-data.chicago_taxi_trips.taxi_trips` table in BigQuery.
2.  **Initial Filtering**: The raw data is filtered based on logical conditions to remove invalid records (e.g., `trip_seconds > 0`, `trip_miles > 0`, `trip_total > 3`).
3.  **Outlier Removal**: The dataset is further cleaned by applying constraints identified during EDA to remove outliers (e.g., capping `trip_hours`, `trip_speed`).
4.  **Feature Engineering**: New features (`trip_hours`, `trip_speed`, `dayofweek`, `hour`) are created as described in the previous section.
5.  **Categorical Data Filtering**: The `payment_type` feature is filtered to include only `Credit Card` and `Cash`.
6.  **Categorical Encoding**: The filtered `payment_type` is label-encoded into a numerical format (0 for `Credit Card`, 1 for `Cash`). The binned `dayofweek` and `hour` features are also encoded numerically.
7.  **Data Splitting**: The final, cleaned feature set (`X`) and target variable (`y`) are split into training (75%) and testing (25%) sets using `train_test_split`.

This pipeline ensures that the model is trained on clean, relevant, and correctly formatted data. For a production system, these steps would be encapsulated in a reusable function or a formal pipeline (e.g., using Kubeflow Pipelines or scikit-learn's `Pipeline` object) that could be called by the model serving component to process new, raw data before prediction.

-----

## 5\. Machine Learning Model Design(s) and Selection

### Which Machine Learning Model/Algorithm(s) Were Chosen

A **Linear Regression** model from the scikit-learn library (`sklearn.linear_model.LinearRegression`) was chosen for this task.

### What Criteria Were Used for Machine Learning Model Selection

The selection of a Linear Regression model was based on the following criteria:

  * **Interpretability**: Linear Regression is a simple, transparent "white-box" model. The learned coefficients directly indicate the magnitude and direction of each feature's influence on the fare, making it easy to understand and explain.
  * **Performance**: The exploratory data analysis (specifically the pair-plot) revealed clear linear relationships between key features like `trip_miles` and `trip_seconds` and the target variable `trip_total`. This suggests that a linear model would be a good fit and likely perform well.
  * **Efficiency**: Linear Regression is computationally inexpensive and fast to train, making it an excellent baseline model and a practical choice for applications where training speed is a consideration.
  * **Project Goal**: A primary objective of the notebook is to demonstrate model deployment and Explainable AI on Vertex AI. A simple yet effective model like Linear Regression serves this purpose well without adding unnecessary complexity from more advanced algorithms.

-----

## 6\. Machine Learning Model Training and Development

### Document the Use of Vertex AI or Kubeflow for Machine Learning Model Training

The model training itself was performed locally within the notebook environment using scikit-learn's `.fit()` method. **Vertex AI** was then used for the critical MLOps stages of **deployment, serving, and explanation**, which are essential parts of the model development lifecycle.

### Dataset Sampling Used for Model Training and Justification

The dataset was sampled using a standard **random split** into training and testing sets.

  * **Training Set**: 75% of the data.
  * **Development/Test Set**: 25% of the data.
    This 75/25 split is a common practice that provides a large enough dataset for the model to learn from while reserving a substantial, independent set of data to validate its performance and ensure it generalizes well to unseen data. The `random_state` parameter was used to ensure the split is reproducible.

### Implementation of Model Training

Model training was implemented in a straightforward manner following Google Cloud best practices for simplicity and reproducibility:

  * The `LinearRegression` model was instantiated.
  * The model was trained using a single call: `reg.fit(X_train, y_train)`.
  * The notebook environment runs on a Vertex AI Workbench managed instance, which provides a secure and scalable environment for development. While this specific model did not require distributed training, Vertex AI provides robust support for it if needed.

### The Model Evaluation Metric and Justification

The primary evaluation metrics used were:

  * **R-squared (R2) Score**: This metric represents the proportion of the variance in the dependent variable that is predictable from the independent variables. An R2 score of 0.93 indicates that the model explains 93% of the variability in taxi fares, which is an excellent fit.
  * **Root Mean Squared Error (RMSE)**: This metric measures the standard deviation of the prediction errors (residuals). A lower RMSE is better, and it is in the same units as the target variable (dollars), making it easy to interpret the average error of the model's predictions.

These metrics are optimal for this business goal because they directly quantify the model's predictive accuracy. A high R2 and low RMSE give the business confidence that the fare predictions are reliable and close to the actual values.

### Hyperparameter Tuning and Model Performance Optimization

For the `LinearRegression` model, there are **no significant hyperparameters to tune**. The model's parameters (the coefficients) are determined analytically during the fitting process. Therefore, a formal hyperparameter tuning step was not necessary. Optimization was focused on feature engineering and data cleaning to provide the best possible input to the model.

### How Bias/Variance Were Determined and Tradeoffs

The bias-variance tradeoff was assessed by comparing the model's performance on the training and testing datasets.

  * **Train R2-score:** 0.932
  * **Test R2-score:** 0.932

The fact that the R2 scores for both the training and test sets are nearly identical and very high indicates that the model has **low bias** (it fits the data well) and **low variance** (it generalizes well to unseen data). There is no evidence of overfitting, where performance on the training set would be significantly higher than on the test set. This strong performance validates the chosen model architecture and the quality of the feature engineering.

-----

## 7\. Machine Learning Model Evaluation

After training, the final model was evaluated on the **independent test dataset** (the 25% of data held back from training). This evaluation provides an unbiased estimate of how the model is expected to perform in a real-world scenario on new, unseen data.

The performance results were as follows:

  * **Test R2-score:** 0.932
  * **Test RMSE:** $3.70

**Interpretation:**
An **R2 score of 0.932** is excellent. It signifies that the model successfully explains over 93% of the variance in the taxi fares in the test set. This demonstrates a very strong predictive capability.

A **Root Mean Squared Error of approximately $3.70** means that, on average, the model's fare predictions deviate from the actual fare by about $3.70. Given the range of fares in the dataset, this level of error is relatively low and suggests that the model's predictions are practically useful for fare estimation.

In summary, the model performs exceptionally well on the unseen test data, confirming its robustness and its readiness for deployment.

-----

## 8\. Model/Application on Google Cloud

### Provide Proof That the Machine Learning Model/Application is Deployed and Served on Google Cloud with Vertex AI

The notebook provides clear proof of the model's deployment and serving on Google Cloud using Vertex AI. The entire process is documented through code:

1.  **Model Upload**: The trained scikit-learn model is saved as a `model.pkl` file, uploaded to a Google Cloud Storage bucket, and then registered as a **Vertex AI Model resource**. Crucially, it is configured with explanation metadata for Vertex Explainable AI.
    ```python
    model = aiplatform.Model.upload(...)
    ```
2.  **Endpoint Creation**: A **Vertex AI Endpoint** is created to serve the model. An endpoint provides a publicly accessible URI for receiving prediction requests.
    ```python
    endpoint = aiplatform.Endpoint.create(...)
    ```
3.  **Model Deployment**: The registered model is deployed to the created endpoint, which provisions the necessary compute resources (`n1-standard-2` machine type) to host the model and handle requests.
    ```python
    model.deploy(endpoint=endpoint, ...)
    ```
4.  **Prediction and Explanation Request**: The final proof of serving is demonstrated by sending a prediction request to the deployed endpoint with sample instances from the test set. The `endpoint.explain()` function is called, which not only returns the model's predictions but also the feature attributions calculated by Vertex Explainable AI.
    ```python
    response = endpoint.explain(instances=test_json)
    ```

The successful return of both predictions and explanations from this call confirms that the model is actively deployed, served, and accessible via the Vertex AI platform.
