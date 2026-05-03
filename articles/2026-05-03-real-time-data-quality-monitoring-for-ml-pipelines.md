---
tags: machine-learning, data-quality, real-time-monitoring, great-expectations
---

# Data Quality in the Fast Lane: Implementing Real-Time Monitoring for ML Pipelines with Great Expectations
![Real-Time Data Quality Monitoring for ML Pipelines](../images/real-time-data-quality-monitoring-for-ml.jpg)

## TL;DR
* Implementing real-time data quality monitoring is crucial for maintaining ML model performance in production environments.
* Great Expectations (GE) is a powerful tool for defining and validating data quality expectations.
* By integrating GE with streaming data platforms, you can ensure data quality in real-time.

## Introduction

As machine learning (ML) pipelines move toward real-time inference and continuous deployment, the importance of real-time data quality monitoring cannot be overstated. Stale or anomalous data can instantly erode model performance, leading to subpar results and potential business losses. In this article, we'll explore the current state of the art in real-time data quality monitoring, with a focus on implementing Great Expectations (GE) for ML pipelines.

## Why Real-Time Data Quality Matters

Traditionally, data quality checks were batch-oriented, occurring before model training or scoring. However, with the shift toward real-time inference and continuous deployment, this approach is no longer sufficient. Real-time data quality monitoring is essential for detecting and addressing data quality issues as they arise.

### Key Breakthroughs

Several key breakthroughs have enabled real-time data quality monitoring:

* **Streaming Data Quality Validation**: GE now supports integration with streaming platforms like Kafka, Spark Streaming, and Flink, enabling validations on micro-batches or event streams.
* **Declarative Data Quality Contracts**: Using GE's YAML/JSON configs, teams can assert expectations on-the-fly, such as column values in a range, uniqueness, and null thresholds.
* **Automated Alerting & Observability**: Integrations with monitoring stacks like Prometheus, Grafana, and OpenTelemetry allow for real-time alerting and observability.

## Technical Deep Dive

Let's dive into the technical details of implementing real-time data quality monitoring with GE. We'll explore a simple example using Python and Spark Structured Streaming.

### Step 1: Define Expectations

First, we define our data quality expectations using GE's Python API:
```python
import great_expectations as ge

# Create a GE context
context = ge.get_context()

# Define a dataset expectation suite
suite = context.create_expectation_suite("my_suite")

# Add expectations to the suite
suite.add_expectation(ge.Expectation(
    expectation_type="expect_column_values_to_be_between",
    kwargs={"column": "user_id", "min_value": 1, "max_value": 1000}
))
suite.add_expectation(ge.Expectation(
    expectation_type="expect_column_values_to_be_unique",
    kwargs={"column": "transaction_id"}
))
```
### Step 2: Validate Streaming Data

Next, we integrate GE with Spark Structured Streaming to validate our streaming data:
```python
from pyspark.sql import SparkSession
from great_expectations.dataset import SparkDFDataset

# Create a SparkSession
spark = SparkSession.builder.appName("Real-Time Data Quality").getOrCreate()

# Define a Spark Structured Streaming source
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "my_topic") \
    .load()

# Create a GE dataset from the Spark DataFrame
ge_dataset = SparkDFDataset(df)

# Validate the dataset against our expectation suite
validation_result = ge_dataset.validate(suite)
```
### Step 3: Handle Validation Results

Finally, we handle the validation results and trigger alerts if necessary:
```python
if not validation_result.success:
    # Trigger an alert using Prometheus or other monitoring tools
    print("Data quality validation failed!")
    # Send alert to monitoring stack
```
## Production Architecture

Our production architecture follows the **Streaming Data Ingestion with Real-Time Validation** pattern:

 Kafka → Spark Structured Streaming → Great Expectations → ML Model Service

Here's a text-based representation of the architecture:
```
          +---------------+
          |  Kafka        |
          +---------------+
                  |
                  |
                  v
+---------------+       +---------------+
| Spark Structured|       |  Great        |
|  Streaming     |       |  Expectations  |
+---------------+       +---------------+
                  |               |
                  |               |
                  v               v
+---------------+       +---------------+
|  ML Model      |       |  Monitoring   |
|  Service       |       |  Stack (Prometheus,|
|                |       |  Grafana, etc.)|
+---------------+       +---------------+
```
## Production Lessons Learned

From our real-world experience, we've learned the following key lessons:

* **Start small**: Begin with a minimal set of expectations and gradually add more as needed.
* **Monitor and adjust**: Continuously monitor your data quality and adjust your expectations accordingly.
* **Integrate with monitoring stacks**: Use tools like Prometheus and Grafana to monitor your data quality and trigger alerts.

## Key Takeaways

* Real-time data quality monitoring is crucial for maintaining ML model performance in production environments.
* Great Expectations is a powerful tool for defining and validating data quality expectations.
* By integrating GE with streaming data platforms, you can ensure data quality in real-time.

## Further Reading

For more information on Great Expectations and real-time data quality monitoring, check out the following resources:

* [Great Expectations Documentation](https://docs.greatexpectations.io/en/latest/)
* [Uber's Real-Time Data Quality Framework](https://eng.uber.com/real-time-data-quality/)
* [Airbnb's Data Quality Engineering](https://www.oreilly.com/library/view/airbnb-engineering-data/9781098124983/ch01.html)

By Sheikh Muhammad Qasim | ML Architect