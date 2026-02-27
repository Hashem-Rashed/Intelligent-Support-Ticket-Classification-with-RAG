# MLOps Pipeline

This document describes the MLOps infrastructure and continuous training pipeline for the Intelligent Support Ticket Classification system.

## Table of Contents

1. [Overview](#overview)
2. [Experiment Tracking](#experiment-tracking)
3. [Model Registry](#model-registry)
4. [Continuous Training](#continuous-training)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Best Practices](#best-practices)

## Overview

The MLOps pipeline involves:

- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Model Registry**: Centralized model versioning and deployment
- **Automated Retraining**: Triggered by data drift or performance degradation
- **Monitoring**: Real-time model performance tracking in production

## Experiment Tracking

### MLflow Setup

```bash
# Start MLflow server
mlflow server -h 0.0.0.0 -p 5000

# Access UI at http://localhost:5000
```

### Logging Experiments

```python
from src.mlops.mlflow_tracking import MLFlowTracker

tracker = MLFlowTracker(experiment_name="ticket-classification")

# Log parameters
tracker.log_params({
    "model_type": "bert-base-uncased",
    "batch_size": 32,
    "learning_rate": 2e-5,
})

# Log metrics
tracker.log_metrics({
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.89,
    "f1_score": 0.895,
})

# Log model
tracker.log_model(model, "bert-classifier")
```

## Model Registry

### Registering Models

```python
import mlflow

# Register model
mlflow.register_model("runs:/run_id/bert-classifier", "ticket-classifier")

# Set stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="ticket-classifier",
    version=1,
    stage="Production"
)
```

### Model Versions

```python
# Load specific model version
model = mlflow.pyfunc.load_model("models:/ticket-classifier/Production")

# Load latest model
model = mlflow.pyfunc.load_model("models:/ticket-classifier/Latest")
```

## Continuous Training

### Automated Retraining Triggers

The system automatically retrains when:

1. **Performance Degradation**: Accuracy drops below threshold (75%)
2. **Data Drift**: New data distribution detected
3. **Scheduled Retraining**: Every 7 days
4. **Manual Trigger**: On-demand retraining

### Retraining Pipeline

```python
from src.mlops.retraining_pipeline import RetrainingPipeline

pipeline = RetrainingPipeline(
    model_name="ticket-classifier",
    retrain_interval_days=7,
    min_accuracy_threshold=0.75,
)

# Check if retraining needed
if pipeline.should_retrain(current_accuracy=0.70):
    pipeline.run_retraining(training_data, training_labels)
    new_accuracy = pipeline.evaluate_new_model(val_data, val_labels)
    
    if new_accuracy > current_accuracy:
        # Deploy new model
        deploy_new_model(new_model_path)
```

### Retraining Script

```bash
# Run retraining
python src/mlops/retrain_model.py \
  --model-name ticket-classifier \
  --data-path data/processed/recent.csv \
  --epochs 5

# Monitor retraining
tail -f logs/retraining.log
```

## Monitoring and Alerting

### Model Monitoring

```python
from src.mlops.monitoring import ModelMonitor

monitor = ModelMonitor("ticket-classifier")

# Log predictions for monitoring
for pred in predictions:
    monitor.log_prediction(
        input_data=pred["text"],
        prediction=pred["class"],
        confidence=pred["confidence"],
        ground_truth=pred.get("label")
    )

# Check for drift
if monitor.detect_drift(threshold=0.1):
    print("Data drift detected! Consider retraining.")
```

### Metrics Dashboard

Key metrics to monitor:

```
Model Performance:
- Accuracy over time
- Precision/Recall by class
- F1-Score trend

Data Quality:
- Input text length distribution
- Class balance
- Missing values

Operations:
- Prediction latency (p50, p95, p99)
- API error rate
- Request throughput
```

### Alert Rules

```yaml
# alerts.yaml
alerts:
  - name: accuracy_drop
    condition: accuracy < 0.75
    notification: email
    
  - name: high_latency
    condition: p99_latency > 5000ms
    notification: slack
    
  - name: data_drift
    condition: drift_score > 0.15
    notification: email, slack
```

## Best Practices

### 1. Experiment Tracking

- Always log hyperparameters and metrics
- Use descriptive run names: `{model}_{data}_{date}`
- Track git commit for reproducibility
- Log data version/hash

```python
import subprocess
import hashlib

# Get git commit
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.log_param("git_commit", commit)

# Log data version
with open("data/processed/data.csv", "rb") as f:
    data_hash = hashlib.sha256(f.read()).hexdigest()
mlflow.log_param("data_version", data_hash)
```

### 2. Model Management

- Use semantic versioning for models
- Keep metadata about training data
- Archive old models (keep last 5 versions)
- Document model limitations

```python
metadata = {
    "training_date": "2024-02-28",
    "data_size": 10000,
    "training_duration_hours": 2.5,
    "framework": "pytorch",
    "limitations": ["English text only", "max length 512 tokens"],
}
mlflow.log_dict(metadata, "metadata.json")
```

### 3. Data Pipeline

- Validate data before retraining
- Track data lineage
- Version datasets
- Monitor data quality metrics

```python
# Data validation
assert len(train_data) > 1000, "Insufficient training data"
assert len(train_data) == len(train_labels), "Data/label mismatch"
assert not any(pd.isna(train_data)), "Missing values in data"
```

### 4. Performance Testing

- Test on holdout sets monthly
- Monitor prediction time
- Track GPU/memory usage
- A/B test new models

```python
# Performance benchmarking
import time

start = time.time()
predictions = model.predict(test_data)
duration = time.time() - start

throughput = len(test_data) / duration
mlflow.log_metric("throughput_samples_per_sec", throughput)
```

## Deployment Pipeline

```
Code Commit
    ↓
[Pre-training Checks]
    ↓
Train Model
    ↓
[Evaluation]
    ├─ If performance > threshold: Register
    └─ Else: Log results, notify team
    ↓
[A/B Testing] (optional)
    ↓
Deploy to Production
    ↓
[Monitoring] → [Detect Drift] → [Trigger Retrain]
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [ML Model Monitoring Best Practices](https://cloud.google.com/architecture/mlops)
- [Continuous Training Patterns](https://cloud.google.com/solutions/continuous-delivery-ml)
