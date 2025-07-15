# -*- coding: utf-8 -*-
"""
chicago_taxi_fare_prediction.ipynb
"""

# Copyright 2025 EliteCloud LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Taxi fare prediction using the Chicago Taxi Trips dataset
  

## Overview

This python code demonstrates analysis, feature selection, model building, and deployment with Explainable AI configured on Vertex AI, using a subset of the Chicago Taxi Trips dataset for taxi-fare prediction.

*Note: This notebook is developed to run in a [Vertex AI Workbench managed notebooks](https://console.cloud.google.com/vertex-ai/workbench/list/managed) instance using the Python (Local) kernel. Some components of this notebook may not work in other notebook environments.*

Learn more about [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction) and [Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview).

### Objective

The goal of this notebook is to provide an overview on Vertex AI features like Explainable AI and BigQuery in Notebooks by trying to solve a taxi fare prediction problem.


This tutorial uses the following Google Cloud ML services and resources:

- Vertex AI model resource
- Vertex AI endpoint resource
- Vertex Explainable AI
-  Cloud Storage
-  BigQuery

The steps performed include:

- Loading the dataset using "BigQuery in Notebooks".
- Performing exploratory data analysis on the dataset.
- Feature selection and preprocessing.
- Building a linear regression model using scikit-learn.
- Configuring the model for Vertex Explainable AI.
- Deploying the model to Vertex AI.
- Testing the deployed model.
- Clean up.

### Dataset

The Chicago Taxi Trips dataset includes taxi trips from 2013 to the present, reported to the city of Chicago in its role as a regulatory agency. To protect privacy but allow for aggregate analyses, the taxi ID is consistent for any given taxi medallion number but does not show the number, census tracts are suppressed in some cases, and times are rounded to the nearest 15 minutes. Due to the data reporting process, not all trips are reported but the city believes that most are. This dataset is publicly available on BigQuery as a public dataset with the table ID `bigquery-public-data.chicago_taxi_trips.taxi_trips` and also as a public dataset on Kaggle at [Chicago Taxi Trips](https://www.kaggle.com/chicago/chicago-taxi-trips-bq).

For more information about this dataset and how it was created, see the [Chicago Digital website](http://digital.cityofchicago.org/index.php/chicago-taxi-data-released).

The original dataset considered for this tutorial is a large and noisy one and so data from a specific date range is used. Based on various online resources, the data from around May 2018 gave some really good results compared to the other date ranges. While there are also some complicated models proposed for the same problem, like considering the weather data, holidays and seasons, the current notebook only explores a simple linear regression model. Our main objective is to demonstrate the model deployment with Vertex Explainable AI configured on Vertex AI.

The chosen dataset consists of the following fields:

- `unique_key` : Unique identifier for the trip.
- `taxi_id` : A unique identifier for the taxi.
- `trip_start_timestamp`: When the trip started, rounded to the nearest 15 minutes.
- `trip_end_timestamp`: When the trip ended, rounded to the nearest 15 minutes.
- `trip_seconds`: Time of the trip in seconds.
- `trip_miles`: Distance of the trip in miles.
- `pickup_census_tract`: The Census Tract where the trip began. For privacy, this Census Tract is not shown for some trips.
- `dropoff_census_tract`: The Census Tract where the trip ended. For privacy, this Census Tract is not shown for some trips.
- `pickup_community_area`: The Community Area where the trip began.
- `dropoff_community_area`: The Community Area where the trip ended.
- `fare`: The fare for the trip.
- `tips`: The tip for the trip. Cash tips are generally not recorded.
- `tolls`: The tolls for the trip.
- `extras`: Extra charges for the trip.
- `trip_total`: Total cost of the trip, the total of the fare, tips, tolls, and extras.
- `payment_type`: Type of payment for the trip.
- `company`: The taxi company.
- `pickup_latitude`: The latitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy.
- `pickup_longitude`: The longitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy.
- `pickup_location`: The location of the center of the pickup census tract or the community area if the census tract has been hidden for privacy.
- `dropoff_latitude`: The latitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy.
- `dropoff_longitude`: The longitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy.
- `dropoff_location`: The location of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy.

### Costs

This tutorial uses the following billable components of Google Cloud:

- Vertex AI
- BigQuery
- Cloud Storage


Learn about [Vertex AI
pricing](https://cloud.google.com/vertex-ai/pricing), [BigQuery pricing](https://cloud.google.com/bigquery/pricing) and [Cloud Storage
pricing](https://cloud.google.com/storage/pricing), and use the [Pricing
Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

### Get started

### Install Vertex AI SDK for Python and other required packages
"""

! pip3 install --quiet --upgrade    google-cloud-bigquery \
                                    google-cloud-aiplatform \
                                    google-cloud-storage \
                                    seaborn==0.12.0 \
                                    numpy==1.26.4 \
                                    scikit-learn \
                                    pandas==2.0.3 \
                                    fsspec==2024.6.0 \
                                    db-dtypes \
                                    pyarrow==14.0.0

"""### Restart runtime (Colab only)

To use the newly installed packages, you must restart the runtime on Google Colab.
"""

import sys

if "google.colab" in sys.modules:

    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)

"""<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

Authenticate your environment on Google Colab.
"""

import sys

if "google.colab" in sys.modules:

    from google.colab import auth

    auth.authenticate_user()

"""### Set Google Cloud project information

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""

PROJECT_ID = "a94-project-ai-specialization"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type: "string"}

"""### UUID

If you are in a live tutorial session, you might be using a shared test account or project. To avoid name collisions between users on resources created, you create a uuid for each instance session, and append it onto the name of resources you create in this tutorial.
"""

import random
import string


# Generate a uuid of length 8
def generate_uuid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


UUID = generate_uuid()

"""### Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.
"""

BUCKET_URI = f"gs://your-bucket-name-{PROJECT_ID}-unique"  # @param {type:"string"}

"""**If your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket."""

! gsutil mb -l $LOCATION -p $PROJECT_ID $BUCKET_URI

"""### Import libraries"""

# Commented out IPython magic to ensure Python compatibility.
import pickle

import matplotlib.pyplot as plt
# load the required libraries
import pandas as pd
import seaborn as sns
from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1.types import SampledShapleyAttribution
from google.cloud.aiplatform_v1.types.explanation import ExplanationParameters
from google.cloud.bigquery import Client

# %matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

"""## Accessing the data through BigQuery Integration

The **BigQuery Integration for Notebooks** feature of Vertex AI Workbench managed notebooks lets you use BigQuery and its features from the notebook itself eliminating the need to switch between tabs everytime. For every cell in the notebook, there is an option for the BigQuery integration at the top right, and selecting it enables you to compose an SQL query that can be executed in BigQuery.


Among the available fields in the dataset, only the fields that seem common and relevant for analysis and modeling like `taxi_id`, `trip_start_timestamp`, `trip_seconds`, `trip_miles`, `payment_type` and `trip_total` are selected. Further, the field `trip_total` is treated as the target variable that would be predicted by the machine learning model. Apparently, this field is a summation of the `fare`,`tips`,`tolls` and `extras` fields and so because of their correlation with the target variable, they are being excluded for modeling. Due to the volume of the data, a subset of the dataset over the course of one week, 12-May-2018 to 18-May-2018 is being considered. Within this date range itself, the datapoints can be noisy and so a few conditions like the following are considered:

- Time taken for the trip > 0.
- Distance covered during the trip > 0.
- Total trip charges > 0 and
- Pickup and dropoff areas are valid (not empty).

Note: The below cell is a Bigquery Integration cell and can only execute on Vertex AI Workbench's managed instances. If your notebook environment is different, you can skip it.

#@bigquery

select
-- select the required fields
taxi_id, trip_start_timestamp,
trip_seconds, trip_miles, trip_total,
payment_type

from `bigquery-public-data.chicago_taxi_trips.taxi_trips`
where
-- specify the required criteria
trip_start_timestamp >= '2018-05-12' and
trip_end_timestamp <= '2018-05-18' and
trip_seconds > 0 and
trip_miles > 0 and
trip_total > 3 and
pickup_community_area is not NULL and
dropoff_community_area is not NULL

The BigQuery integration also lets you load the queried data into a pandas dataframe using the `Query and load as DataFrame` button. Clicking the button adds a new cell below that provides a code snippet to load the data into a dataframe.

#### Select the required fields
"""

# The following two lines are only necessary to run once.
# Comment out otherwise for speed-up.

client = Client(project=PROJECT_ID)

query = """select
taxi_id, trip_start_timestamp,
trip_seconds, trip_miles, trip_total,
payment_type, pickup_community_area,
dropoff_community_area

from `bigquery-public-data.chicago_taxi_trips.taxi_trips`
where
trip_start_timestamp >= '2018-05-12' and
trip_end_timestamp <= '2018-05-18' and
trip_seconds > 60 and trip_seconds < 6*60*60 and
trip_miles > 0 and
trip_total > 3 and
pickup_community_area is not NULL and
dropoff_community_area is not NULL"""
job = client.query(query)
df = job.to_dataframe()

"""#### Check the fields in the data and their shape."""

# check the dataframe's shape
print(df.shape)
# check the columns in the dataframe
df.columns

"""#### Check some sample data."""

df.head()

"""#### Check the dtypes of fields in the data."""

df.dtypes

"""#### Check for null values in the dataframe."""

df.info()

"""Depending on the percentage of null values in the data, one can choose to either drop them or impute them with mean/median (for numerical values) and mode (for categorical values). In the current data, there doesn't seem to be any null values.

#### Check the numerical distributions of the fields (numerical).

In case there are any fields with constant values, those fields can be dropped as they don't add any value to the model.
"""

df.describe().T

"""#### Identify the categorical and numerical fields in the data

In the current dataset, `trip_total` is the target field. To access the fields by their type easily, identify the categorical and numerical fields in the data and save them.
"""

target = "trip_total"
categ_cols = ["payment_type", "pickup_community_area", "dropoff_community_area"]
num_cols = ["trip_seconds", "trip_miles"]

"""## Analyze numerical data

To further anaylyze the data, there are various plots that can be used on numerical and categorical fields. In case of numerical data, you can use histograms and box plots. Bar charts are suited for categorical data to better understand the distribution of the data and the outliers in the data.

#### Plot histograms and box plots on the numerical fields.
"""

for i in num_cols + [target]:
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    df[i].plot(kind="hist", bins=100, ax=ax[0])
    ax[0].set_title(str(i) + " -Histogram")
    df[i].plot(kind="box", ax=ax[1])
    ax[1].set_title(str(i) + " -Boxplot")
    plt.show()

"""#### The field `trip_seconds` describes the time taken for the trip in seconds. For ease of our analysis, let us convert it into hours."""

df["trip_hours"] = round(df["trip_seconds"] / 3600, 2)
df["trip_hours"].plot(kind="box")

"""#### Similarly, another field `trip_speed` can be added by dividing `trip_miles` and `trip_hours` to understand the speed of the trip in miles/hour."""

df["trip_speed"] = round(df["trip_miles"] / df["trip_hours"], 2)
df["trip_speed"].plot(kind="box")

"""#### So far you've only looked at the univariate plots. To better understand the relationship between the variables, a pair-plot can be plotted."""

# generate a pairplot for 10K samples
try:
    sns.pairplot(
        data=df[["trip_seconds", "trip_miles", "trip_total", "trip_speed"]].sample(
            10000
        )
    )
    plt.show()
except Exception as e:
    print(e)

"""From the box plots and the histograms visualized so far, it is evident that there are some outliers causing skewness in the data which perhaps could be removed. Also, you can see some linear relationships between the independent variables considered in the pair-plot. For example, `trip_seconds` and `trip_miles` and the dependant variable `trip_total`.

#### Restrict the data based on the following conditions to remove the outliers in the data to some extent :
- Total charge being at least more than $3.
- Total miles driven greater than 0 and less than 300 miles.
- Total seconds driven at least 1 minute.
- Total hours driven not more than 2 hours.
- Speed of the trip not being more than 70 mph.

These conditions are based on some general assumptions as clearly there were some recording errors like speed being greater than 500 mph and travel-time being more than 5 hours that led to outliers in the data.
"""

# set constraints to remove outliers
df = df[df["trip_total"] > 3]

df = df[(df["trip_miles"] > 0) & (df["trip_miles"] < 300)]

df = df[df["trip_seconds"] >= 60]

df = df[df["trip_hours"] <= 2]

df = df[df["trip_speed"] <= 70]
df.reset_index(drop=True, inplace=True)
df.shape

"""## Analyze categorical data

#### Further, explore the categorical data by plotting the distribution of all the levels in each field.
"""

for i in categ_cols:
    print(df[i].unique().shape)
    df[i].value_counts(normalize=True).plot(kind="bar", figsize=(10, 4))
    plt.title(i)
    plt.show()

"""From the above analysis, one can see that almost 99% of the transaction types are Cash and Credit Card. While there are also other type of transactions, their distribution is negligible. In such a case, the lower distribution levels can be dropped. On the other hand, the total number of pickup and dropoff community areas both seem to have the same levels which make sense. In this case also, one can choose to omit the lower distribution levels but you'd have to make sure that both the fields have the same levels afterward. In the current notebook, keep them as is and proceed with the modeling.

The relationships between the target variable and the categorical fields can be represented through box plots. For each level, the corresponding distribution of the target variable can be identified.
"""

for i in categ_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=i, y=target, data=df)
    plt.xticks(rotation=45)
    plt.title(i)
    plt.show()

"""There seems to be one case where the `trip_total` is over 3000 and has the same pickup and dropoff community area: 28 is clearly an outlier compared to the rest of the points. This datapoint can be removed."""

df = df[df["trip_total"] < 3000].reset_index(drop=True)

"""#### Keep only the `Credit Card` and `Cash` payment types. Further, encode them by assigning 0 for `Credit Card` and 1 for `Cash` payment types."""

# add payment_type
df = df[df["payment_type"].isin(["Credit Card", "Cash"])].reset_index(drop=True)
# encode the payment types
df["payment_type"] = df["payment_type"].apply(
    lambda x: 0 if x == "Credit Card" else (1 if x == "Cash" else None)
)

"""#### There are also useful timestamp fields in the data. `trip_start_timestamp` represents the start timestamp of the taxi trip and fields like what day of week it was and what hour it was can be derived from it."""

df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"])
df["dayofweek"] = df["trip_start_timestamp"].dt.dayofweek
df["hour"] = df["trip_start_timestamp"].dt.hour

"""Since the current dataset is limited to only a week, if there isn't much variation in the newly derived fields with respect to the target variable, they can be dropped.

#### Plot sum and average of the `trip_total` with respect to the `dayofweek`.
"""

# plot sum and average of trip_total w.r.t the dayofweek
_, ax = plt.subplots(1, 2, figsize=(10, 4))
df[["dayofweek", "trip_total"]].groupby("dayofweek").trip_total.sum().plot(
    kind="bar", ax=ax[0]
)
ax[0].set_title("Sum of trip_total")
df[["dayofweek", "trip_total"]].groupby("dayofweek").trip_total.mean().plot(
    kind="bar", ax=ax[1]
)
ax[1].set_title("Avg. of trip_total")
plt.show()

"""#### Plot sum and average of the `trip_total` with respect to the `hour`."""

_, ax = plt.subplots(1, 2, figsize=(10, 4))
df[["hour", "trip_total"]].groupby("hour").trip_total.sum().plot(kind="bar", ax=ax[0])
ax[0].set_title("Sum of trip_total")
df[["hour", "trip_total"]].groupby("hour").trip_total.mean().plot(kind="bar", ax=ax[1])
ax[1].set_title("Avg. of trip_total")
plt.show()

"""As these plots don't seem to have constant figures with respect to the target variable across their levels, they can be considered for training. In fact, to simplify things these derived features can be bucketed into fewer levels.

The `dayofweek` field can be bucketed into a binary field considering whether or not it was a weekend. If it is a weekday, the record can be assigned 1, else 0. Similarly, the `hour` field can also be bucketed and encoded. The normal working hours in Chicago can be assumed to be between *8AM*-*10PM* and if the value falls in between the working hours, it can be encoded as 1, else 0.
"""

# bucket and encode the dayofweek and hour
df["dayofweek"] = df["dayofweek"].apply(lambda x: 0 if x in [5, 6] else 1)
df["hour"] = df["hour"].apply(lambda x: 0 if x in [23, 0, 1, 2, 3, 4, 5, 6, 7] else 1)

"""#### Check the data distribution before training the model."""

df.describe().T

"""## Divide the data into train and test sets

#### Split the preprocessed dataset into train and test sets so that the linear regression model can be validated on the test set.
"""

cols = [
    "trip_seconds",
    "trip_miles",
    "payment_type",
    "pickup_community_area",
    "dropoff_community_area",
    "dayofweek",
    "hour",
    "trip_speed",
]
x = df[cols].copy()
y = df[target].copy()

# split the data into 75-25% ratio
X_train, X_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, test_size=0.25, random_state=13
)
X_train.shape, X_test.shape

"""## Fit a simple linear regression model

#### Fit a linear regression model using scikit-learn's LinearRegression method on the train data.
"""

# Building the regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

"""#### Print the `R2 score` and `RMSE` values for the model on train and test sets."""

# print test R2 score
y_train_pred = reg.predict(X_train)
train_score = r2_score(y_train, y_train_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
y_test_pred = reg.predict(X_test)
test_score = r2_score(y_test, y_test_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)
print("Train R2-score:", train_score, "Train RMSE:", train_rmse)
print("Test R2-score:", test_score, "Test RMSE:", test_rmse)

"""A low RMSE error and a train and test R2 score of 0.93 suggests that the model is fitted well. Further, the coefficients learned by the model for each of its independent variables can also be checked by checking the `coef_` attribute of the sklearn model.

#### Check the coefficients learned by the model.
"""

coef_df = pd.DataFrame({"col": cols, "coeff": reg.coef_})
coef_df.set_index("col").plot(kind="bar")

"""## Save the model and upload to a Cloud Storage bucket

#### To deploy the model on Vertex AI, the model artifacts need to be stored in a Cloud Storage bucket first.
"""

FILE_NAME = "model.pkl"
with open(FILE_NAME, "wb") as file:
    pickle.dump(reg, file)

# Upload the saved model file to Cloud Storage
BLOB_PATH = "taxicab_fare_prediction/"

BLOB_NAME = BLOB_PATH + FILE_NAME

bucket = storage.Client().bucket(BUCKET_URI[5:])
blob = bucket.blob(BLOB_NAME)
blob.upload_from_filename(FILE_NAME)

"""## Deploy the model on Vertex AI with support for Vertex Explainable AI

Configure Vertex Explainable AI before deploying the model. Learn more about [Configuring Vertex Explainable AI in Vertex AI models](https://cloud.google.com/vertex-ai/docs/explainable-ai/configuring-explanations#scikit-learn-and-xgboost-pre-built-containers).
"""

MODEL_DISPLAY_NAME = "demo4_taxi_fare_prediction_model"  # @param {type: "string"}

# If the model display name is not set, choose the default one
if MODEL_DISPLAY_NAME == "[your-model-display-name]":
    MODEL_DISPLAY_NAME = "taxi_fare_prediction_model"


ARTIFACT_GCS_PATH = f"{BUCKET_URI}/{BLOB_PATH}"

# Feature-name(Inp_feature) and Output-name(Model_output) can be arbitrary
exp_metadata = {"inputs": {"Input_feature": {}}, "outputs": {"Predicted_taxi_fare": {}}}

"""#### Create a model resource from the uploaded model with explanation metadata configured."""

# Create a Vertex AI model resource with support for Vertex Explainable AI

aiplatform.init(project=PROJECT_ID, location=LOCATION)

model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=ARTIFACT_GCS_PATH,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    explanation_metadata=exp_metadata,
    explanation_parameters=ExplanationParameters(
        sampled_shapley_attribution=SampledShapleyAttribution(path_count=25)
    ),
)

model.wait()

print(model.display_name)
print(model.resource_name)

"""### Create an Endpoint resource for the model

#### Set a display name for the endpoint and create the endpoint resource.
"""

ENDPOINT_DISPLAY_NAME = "demo4_taxi_fare_prediction_endpoint"  # @param {type: "string"}

# If the display name is not set, choose the default one
if ENDPOINT_DISPLAY_NAME == "[your-endpoint-display-name]":
    ENDPOINT_DISPLAY_NAME = "taxi_fare_prediction_endpoint"

endpoint = aiplatform.Endpoint.create(
    display_name=ENDPOINT_DISPLAY_NAME, project=PROJECT_ID, location=LOCATION
)

print(endpoint.display_name)
print(endpoint.resource_name)

"""### Deploy the model to the created endpoint with the required machine type

#### Set a name for the deployment and deploy the model to the created endpoint.
"""

DEPLOYED_MODEL_NAME = "demo_taxi_fare_prediction_deployment"  # @param {type: "string"}

# If the deployment name is not set, choose the default one
if DEPLOYED_MODEL_NAME == "[your-deployed-model-name]":
    DEPLOYED_MODEL_NAME = "taxi_fare_prediction_deployment"

# Set the machine type to n1-standard2
MACHINE_TYPE = "n1-standard-2"

# Deploy the model to the endpoint
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=DEPLOYED_MODEL_NAME,
    machine_type=MACHINE_TYPE,
)

model.wait()

print(model.display_name)
print(model.resource_name)

"""#### To ensure the model is deployed, the ID of the deployed model can be checked using the `endpoint.list_models()` method."""

endpoint.list_models()

"""## Get explanations from the deployed model

#### For testing the deployed online model, select two instances from the test data as payload.
"""

# format the top 2 test instances as the request's payload
test_json = {"instances": [X_test.iloc[0].tolist(), X_test.iloc[1].tolist()]}

"""Call the endpoint with the payload request and parse the response for explanations. The explanations consists of attributions on the independent variables used for training the model which are based on the configured attribution method. In this case, we've used the `Sampled Shapely` method which assigns credit for the outcome to each feature, and considers different permutations of the features. This method provides a sampling approximation of exact Shapely values. Further information on the attribution methods for explanations can be found at [Overview of Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview)."""

features = X_train.columns.to_list()


def plot_attributions(attrs):
    """
    Function to plot the features and their attributions for an instance
    """
    rows = {"feature_name": [], "attribution": []}
    for i, val in enumerate(features):
        rows["feature_name"].append(val)
        rows["attribution"].append(attrs["Input_feature"][i])
    attr_df = pd.DataFrame(rows).set_index("feature_name")
    attr_df.plot(kind="bar")
    plt.show()
    return


def explain_tabular_sample(
    project: str, location: str, endpoint_id: str, instances: list
):
    """
    Function to make an explanation request for the specified payload and generate feature attribution plots
    """
    aiplatform.init(project=project, location=location)

    # endpoint = aiplatform.Endpoint(endpoint_id)

    response = endpoint.explain(instances=instances)
    print("#" * 10 + "Explanations" + "#" * 10)
    for explanation in response.explanations:
        print(" explanation")
        # Feature attributions.
        attributions = explanation.attributions

        for attribution in attributions:
            print("  attribution")
            print("   baseline_output_value:", attribution.baseline_output_value)
            print("   instance_output_value:", attribution.instance_output_value)
            print("   output_display_name:", attribution.output_display_name)
            print("   approximation_error:", attribution.approximation_error)
            print("   output_name:", attribution.output_name)
            output_index = attribution.output_index
            for output_index in output_index:
                print("   output_index:", output_index)

            plot_attributions(attribution.feature_attributions)

    print("#" * 10 + "Predictions" + "#" * 10)
    for prediction in response.predictions:
        print(prediction)

    return response


test_json = [X_test.iloc[0].tolist(), X_test.iloc[1].tolist()]
prediction = explain_tabular_sample(PROJECT_ID, LOCATION, endpoint, test_json)

"""## Next steps

Since the Chicago Taxi Trips dataset is continuously updating, one can preform the same kind of analysis and model training every time a new set of data is available. The date range can also be increased from a week to a month or more depending on the quality of the data. Most of the steps followed in this notebook would still be valid and can be applied over the new data unless the data is too noisy. In fact, the notebook itself can be scheduled to run at the specified times to retrain the model using the scheduling option of [Vertex AI Workbench's executor](https://console.cloud.google.com/vertex-ai/workbench/list/executions).

## Clean up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial:

- Model
- Endpoint
- Cloud Storage Bucket
"""

# Undeploy the model
endpoint.undeploy_all()

# Delete the endpoint resource
endpoint.delete()

# Delete the model resource.
model.delete()

# Set this to true only if you'd like to delete your bucket
delete_bucket = False

if delete_bucket:
    ! gsutil -m rm -r $BUCKET_URI
