# Requirements Specification

## 1. Document Purpose

This document defines the functional, non-functional, technical, security, data, testing, and Machine Learning requirements of the **AI Predictive Maintenance Platform**.

Its purpose is to establish what the system must do, under which conditions it must operate, and which criteria will be used to determine whether a functionality has been implemented correctly.

Each requirement is identified using a unique code to facilitate traceability during development, testing, and documentation.

---

# 2. Scope

The first version of the platform will focus on predictive maintenance through **binary classification**.

The system will receive operational machine data and determine whether a failure risk exists.

The initial dataset will be the **AI4I 2020 Predictive Maintenance Dataset**.

The platform must support multiple Machine Learning models, compare their performance, and allow the system or the user to select which model should be used during inference.

During the first two project phases, the following areas will be implemented:

- data processing;
- model training;
- model evaluation;
- inference;
- model management;
- API development;
- persistence;
- security;
- audit logging;
- experiment tracking;
- user interface;
- containerization.

---

# 3. System Actors

## ACT-01 — Administrator

A user with full access to the platform.

The Administrator will be allowed to:

- manage users;
- view models;
- modify model status;
- promote models;
- perform inference;
- review audit logs;
- review model metrics.

---

## ACT-02 — ML Engineer

A user responsible for the Machine Learning lifecycle.

The ML Engineer will be allowed to:

- train models;
- evaluate models;
- register experiments;
- review metrics;
- register new model versions;
- promote model candidates when authorized;
- perform inference.

---

## ACT-03 — Operator

A user responsible for using the prediction system.

The Operator will be allowed to:

- enter machine data;
- perform inference;
- select authorized models;
- review prediction results;
- review permitted prediction history.

The Operator will not be allowed to modify critical model configurations.

---

## ACT-04 — Viewer

A user with primarily read-only access.

The Viewer will be allowed to:

- review metrics;
- view registered models;
- review permitted prediction results.

The Viewer will not be allowed to train, modify, or promote models.

---

# 4. Functional Requirements

## FR-01 — Dataset Loading

The system must be able to load the AI4I dataset from a controlled local source.

### Acceptance Criteria

- The original dataset must not be modified during loading.
- The system must detect when the file does not exist.
- File reading errors must be handled gracefully.
- The loaded dataset must preserve the expected number of rows and columns.

---

## FR-02 — Data Validation

The system must validate the dataset structure before using it.

The validation process must verify at least:

- required columns;
- data types;
- missing values;
- duplicated records;
- values outside reasonable ranges;
- presence of the target variable.

### Acceptance Criteria

If a mandatory column is missing, the system must stop the pipeline and return a descriptive error.

---

## FR-03 — Data Preprocessing

The system must transform the data before training and inference.

The preprocessing pipeline may include:

- removal of irrelevant columns;
- categorical encoding;
- numerical feature scaling when required by the model;
- data cleaning;
- feature transformation.

### Acceptance Criteria

The same preprocessing pipeline used during training must also be used during inference.

---

## FR-04 — Dataset Splitting

The system must split the dataset into independent subsets.

Initially:

```text
Training
Validation
Test
```

The split must use a reproducible random seed.

### Acceptance Criteria

The same configuration must generate the same dataset split.

---

## FR-05 — Logistic Regression Training

The system must allow a Logistic Regression model to be trained.

The model must implement the common model interface defined by the platform.

---

## FR-06 — Decision Tree Training

The system must allow a Decision Tree model to be trained.

---

## FR-07 — Random Forest Training

The system must allow a Random Forest model to be trained.

---

## FR-08 — XGBoost Training

The system must allow an XGBoost model to be trained.

---

## FR-09 — Common Model Interface

All models must expose equivalent operations.

At minimum:

```text
train()

predict()

predict_proba()

save()

load()
```

When applicable:

```text
evaluate()
```

### Acceptance Criteria

A high-level system component must be able to work with different models without requiring model-specific changes.

---

## FR-10 — Model Evaluation

The system must evaluate trained models.

Initial metrics must include:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- Confusion Matrix.

### Acceptance Criteria

Evaluation results must be associated with the corresponding model name and version.

---

## FR-11 — Model Comparison

The system must allow multiple models to be compared.

The comparison must show at least:

- model name;
- model version;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

Optional operational metrics may include:

- training time;
- inference latency;
- model size.

---

## FR-12 — Trained Model Persistence

The system must be able to save trained models so they can be reused without retraining.

### Acceptance Criteria

A saved model must be loadable and must produce predictions consistent with the original trained version.

---

## FR-13 — Model Versioning

Each registered model must have an identifiable version.

Example:

```text
random_forest_v1
random_forest_v2
xgboost_v1
```

Each version must be associated with:

- training date;
- configuration;
- dataset;
- metrics;
- model status.

---

## FR-14 — Model Factory

The system must provide a centralized mechanism for creating models from an identifier.

Example:

```text
ModelFactory.create("random_forest")
```

### Acceptance Criteria

Model creation logic must not be duplicated across unrelated components.

---

## FR-15 — Model Manager

The system must include a component responsible for managing available models.

It must support:

- listing models;
- loading models;
- registering versions;
- reviewing model status;
- retrieving the primary production model.

---

## FR-16 — Manual Model Selection

An authorized user must be able to manually select which model should be used during inference.

Examples:

```text
logistic_regression

decision_tree

random_forest

xgboost
```

---

## FR-17 — Automatic Model Selection

The system must provide an automatic selection option:

```text
AUTO
```

In the initial version, AUTO must use the model currently defined as the primary model or Champion.

---

## FR-18 — Model Router

The system must include a Model Router responsible for directing an inference request to the correct model.

The router must receive at least:

- requested model;
- input data.

It must return:

- prediction;
- probability when available;
- model used;
- model version.

---

## FR-19 — Inference

The system must perform inference on new machine data.

Conceptual input:

```text
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
Product type
```

Conceptual output:

```text
Prediction

Probability

Model used

Model version
```

---

## FR-20 — Inference Request Validation

The system must validate input data before performing a prediction.

It must reject:

- missing fields;
- invalid data types;
- incompatible values;
- non-existing models.

---

## FR-21 — Health Check API

The API must provide an endpoint that reports system health.

Example:

```text
GET /health
```

Expected conceptual response:

```text
status: healthy
```

---

## FR-22 — Model Listing Endpoint

The API must allow users to retrieve the list of available models.

Example:

```text
GET /models
```

---

## FR-23 — Prediction Endpoint

The platform must provide an endpoint for inference.

Example:

```text
POST /predict
```

---

## FR-24 — Model Metrics Endpoint

The API must allow model metrics to be retrieved.

Example:

```text
GET /models/metrics
```

---

## FR-25 — Database

During Phase 2, the platform must use PostgreSQL for persistent data storage.

---

## FR-26 — Prediction History

Each inference must be eligible for storage in the system.

At minimum, the following information must be stored:

```text
prediction_id

timestamp

model_name

model_version

prediction

probability

latency
```

When appropriate, the original input may also be stored according to privacy and data-handling policies.

---

## FR-27 — Model Metadata Storage

The database must store metadata related to registered models.

This may include:

- name;
- version;
- status;
- metrics;
- creation date;
- artifact location.

---

## FR-28 — Experiment Tracking

The platform must record Machine Learning experiments.

MLflow is the initial planned experiment tracking tool.

Each experiment should be able to store:

- model;
- parameters;
- hyperparameters;
- metrics;
- artifacts;
- date;
- dataset used.

---

## FR-29 — Model Statuses

Models may have lifecycle states such as:

```text
Experimental

Candidate

Champion

Archived
```

---

## FR-30 — Champion Model

The system must allow one model to be defined as Champion.

In the first version, only one active Champion must exist for each predictive problem.

---

## FR-31 — Challenger Models

The system must support identifying candidate models as Challengers.

Challenger models may be compared against the Champion before promotion.

---

## FR-32 — Model Promotion

An authorized user must be able to promote a candidate model to Champion.

The promotion operation must generate an audit event.

---

## FR-33 — Authentication

Phase 2 must include user authentication.

The system must verify user identity before allowing access to protected functionality.

---

## FR-34 — Password Hashing

Passwords must never be stored in plain text.

The system must only store secure password hashes.

---

## FR-35 — Access Tokens

The platform may use JWT access tokens to manage API sessions.

---

## FR-36 — Role-Based Access Control

The platform must implement RBAC.

Initial roles:

```text
ADMIN

ML_ENGINEER

OPERATOR

VIEWER
```

---

## FR-37 — Authorization

Protected operations must verify user permissions before execution.

Example:

```text
Viewer → View metrics → Allowed

Viewer → Promote model → Denied
```

---

## FR-38 — Audit Logging

The system must record relevant actions.

Examples include:

- login;
- user creation;
- model training;
- model registration;
- model status changes;
- Champion promotion;
- authorization failures.

---

## FR-39 — Dashboard

Phase 2 must include a basic visual user interface.

The dashboard must allow users to:

- enter machine data;
- select a model;
- execute inference;
- view prediction results;
- view prediction probability;
- view available models;
- review primary model metrics.

---

## FR-40 — Containerization

The main system services must be executable using Docker.

At minimum:

```text
Backend

PostgreSQL

MLflow
```

The frontend may also be containerized if implemented as a separate application.

---

# 5. Machine Learning Requirements

## MLR-01 — Proper Dataset Separation

Test data must never be used during model training or hyperparameter selection.

---

## MLR-02 — Data Leakage Prevention

Any preprocessing component that learns parameters from data must be fitted only using the training set.

Example:

```text
Scaler.fit(training_data)
```

and later:

```text
Scaler.transform(validation_data)
Scaler.transform(test_data)
```

---

## MLR-03 — Reproducibility

Experiments must use configurable random seeds whenever possible.

---

## MLR-04 — Baseline Model

A baseline model must exist before more complex models are introduced.

The initial baseline will be:

```text
Logistic Regression
```

---

## MLR-05 — Class Imbalance

Because machine failures may represent a minority class, the system must analyze class imbalance.

Possible strategies may include:

- class weights;
- threshold adjustment;
- resampling when justified.

Accuracy must not be treated as the only meaningful metric.

---

## MLR-06 — Recall Priority for Failure Detection

Special attention must be given to Recall for the positive failure class.

A false negative may represent a real failure that the system did not detect.

---

## MLR-07 — Fair Model Comparison

Models being compared must use:

- the same dataset split;
- the same test set;
- consistent evaluation metrics.

---

## MLR-08 — Experiment Traceability

Every relevant model should be traceable to:

```text
dataset version

preprocessing version

hyperparameters

metrics

code version
```

when this information is available.

---

## MLR-09 — Preprocessing Persistence

All transformers required during inference must be stored together with the relevant model artifacts.

---

## MLR-10 — Configurable Classification Threshold

Probabilistic models must support a configurable classification threshold.

The initial default may be:

```text
0.50
```

but it may later be changed according to evaluation results.

---

# 6. Non-Functional Requirements

## NFR-01 — Modularity

The system must separate components according to responsibility.

Example modules:

```text
data

preprocessing

models

training

evaluation

inference

api

security

database
```

---

## NFR-02 — Low Coupling

Changing one model implementation must not require modifications across unrelated modules.

---

## NFR-03 — High Cohesion

Each module should focus on one clearly defined responsibility.

---

## NFR-04 — Maintainability

The codebase should favor:

- descriptive naming;
- small functions;
- documentation;
- type hints when useful;
- explicit error handling;
- separation of concerns.

---

## NFR-05 — Extensibility

The architecture must allow new models to be added without redesigning the platform.

Future examples:

```text
LSTM

GRU

1D CNN

Autoencoder
```

---

## NFR-06 — Configurability

Configurable values must not be unnecessarily distributed as hardcoded constants throughout the codebase.

Configuration files or environment variables should be used when appropriate.

---

## NFR-07 — Error Handling

The system must handle errors in a structured way.

Possible exceptions include:

```text
DatasetNotFoundError

MissingFeatureError

InvalidModelError

ModelNotLoadedError

UnauthorizedError
```

---

## NFR-08 — Logging

The platform must generate logs for relevant events.

At minimum:

- service startup;
- errors;
- model loading;
- training;
- failed inference requests;
- model status changes.

---

## NFR-09 — Performance

A single inference request using the initial classical Machine Learning models should complete fast enough for real-time interaction from the user interface.

Initial target:

```text
< 1 second
```

under a reasonable local development environment.

---

## NFR-10 — API Reliability

The API must respond gracefully to invalid input.

An invalid user request must not terminate the application process.

---

## NFR-11 — Compatibility

The project must run on a modern Python environment and document its dependencies.

---

## NFR-12 — Environment Reproducibility

Dependencies must be explicitly declared using files such as:

```text
requirements.txt
```

or:

```text
pyproject.toml
```

---

## NFR-13 — Documentation

Main project functionality must be documented inside:

```text
docs/
```

The root README must provide enough information to install and run the project.

---

# 7. Security Requirements

## SEC-01 — No Plain-Text Password Storage

Passwords must never be stored in plain text.

---

## SEC-02 — Secrets Outside Source Code

Credentials, tokens, and secret keys must be stored using environment variables or another appropriate secrets-management mechanism.

Real secrets must never be committed to the repository.

---

## SEC-03 — `.env` File Protection

The real:

```text
.env
```

file must be excluded from Git.

The project may include:

```text
.env.example
```

without real secrets.

---

## SEC-04 — Input Validation

All external input must be validated before reaching critical system components.

---

## SEC-05 — Endpoint Authorization

Protected endpoints must verify the user's role before allowing execution.

---

## SEC-06 — Principle of Least Privilege

Users must only receive the permissions necessary for their responsibilities.

---

## SEC-07 — Critical Action Auditing

Operations such as model promotion or deletion must generate an audit record.

---

## SEC-08 — Safe Error Responses

External error responses must not expose sensitive information such as:

- passwords;
- tokens;
- unnecessary internal paths;
- production stack traces.

---

## SEC-09 — Rate Limiting

The platform should be prepared to support request rate limiting.

It may be implemented during Phase 2 if time allows.

---

## SEC-10 — HTTPS

Public deployments must use HTTPS.

HTTP may be used during local development.

---

# 8. Data Requirements

## DR-01 — Immutable Raw Dataset

Files stored inside:

```text
data/raw/
```

must not be modified by the processing pipeline.

---

## DR-02 — Processed Data Separation

Transformed datasets must be stored inside:

```text
data/processed/
```

or be reproducibly generated by the processing pipeline.

---

## DR-03 — Dataset Version Identification

When multiple processed dataset versions exist, they must be identifiable.

---

## DR-04 — Expected Feature Schema

The system must maintain a clear definition of the features expected by each model.

---

## DR-05 — Range Validation

The system should identify physically or logically impossible values whenever sufficient domain information is available.

---

## DR-06 — Training and Inference Consistency

The feature schema used during inference must match the feature schema used during training.

---

# 9. Testing Requirements

## TR-01 — Unit Tests

Unit tests must exist for critical isolated components.

Examples:

```text
data loader

validator

preprocessing

metrics

model factory

model router
```

---

## TR-02 — Integration Tests

Integration tests must validate complete workflows.

Example:

```text
Load data
   ↓
Preprocess
   ↓
Train
   ↓
Predict
```

---

## TR-03 — API Tests

Primary endpoints must be tested for:

- valid responses;
- invalid input;
- missing fields;
- invalid model identifiers.

---

## TR-04 — Security Tests

Basic permissions must be tested.

Example:

```text
VIEWER
   ↓
POST /models/promote
   ↓
403 Forbidden
```

---

## TR-05 — Model Tests

Models must pass basic validation tests.

Predictions:

```text
predictions ∈ {0, 1}
```

Probabilities:

```text
0 ≤ probability ≤ 1
```

---

# 10. Project Constraints

## PC-01 — Time

The first two project phases will have an approximate total duration of:

```text
2 months
```

---

## PC-02 — Individual Development

The project will primarily be developed by one person.

Therefore, maintainability and realistic scope will take priority over unnecessary distributed-system complexity.

---

## PC-03 — MVP Before Advanced Features

Core functionality must be completed before advanced technologies are added.

---

## PC-04 — Technologies Outside the Initial MVP

The following technologies are not mandatory during the first two phases:

- Kubernetes;
- Kafka;
- distributed microservices;
- LSTM;
- GRU;
- Transformers;
- ESP32;
- MQTT;
- Service Mesh;
- SPIFFE/SPIRE;
- complex cloud infrastructure.

They may be added in future development phases.

---

# 11. Planned Technology Stack

The technology stack may change if there is a valid technical justification.

The initial stack is:

```text
Language:
Python

Machine Learning:
scikit-learn
XGBoost

Data Processing:
pandas
NumPy

API:
FastAPI

Validation:
Pydantic

Database:
PostgreSQL

ORM:
SQLAlchemy

Experiment Tracking:
MLflow

Testing:
pytest

Containers:
Docker
Docker Compose

Version Control:
Git
GitHub
```

For the user interface, the initial candidates are:

```text
React
```

or a simpler alternative if project time constraints require it.

---

# 12. Requirement Prioritization

Requirements will be classified into three priority levels.

## MUST

Required for the initial version to be considered functional.

Examples:

```text
Data pipeline

Multiple models

Model evaluation

Model selection

Inference

API

Basic testing
```

---

## SHOULD

Important requirements that make the prototype closer to a professional platform.

Examples:

```text
PostgreSQL

MLflow

Authentication

RBAC

Audit logging

Docker
```

---

## COULD

Optional functionality that may be implemented if enough time is available.

Examples:

```text
Advanced dashboard

Rate limiting

Automated model promotion

Data drift detection

Advanced monitoring
```

---

# 13. Minimum Requirements for Phase 1 Completion

Phase 1 will be considered complete when at least the following requirements have been implemented:

```text
FR-01  Dataset loading

FR-02  Data validation

FR-03  Preprocessing

FR-04  Dataset splitting

FR-05  Logistic Regression

FR-06  Decision Tree

FR-07  Random Forest

FR-08  XGBoost

FR-09  Common model interface

FR-10  Model evaluation

FR-11  Model comparison

FR-12  Model persistence

FR-14  Model Factory

FR-15  Model Manager

FR-16  Manual model selection

FR-17  AUTO selection

FR-18  Model Router

FR-19  Inference

FR-20  Request validation

FR-21  Health endpoint

FR-22  Models endpoint

FR-23  Prediction endpoint

FR-24  Metrics endpoint
```

Basic automated tests for the main workflow must also exist.

---

# 14. Minimum Requirements for Phase 2 Completion

Phase 2 will be considered complete when, in addition to Phase 1, the following functionality has been implemented:

```text
PostgreSQL

Prediction history

Model metadata

MLflow experiment tracking

Model versioning

Champion / Challenger

Authentication

RBAC

Audit logging

Basic dashboard

Docker

Integration tests
```

---

# 15. Definition of Done

A functionality will not be considered complete only because it works during a manual test.

When applicable, a completed feature must include:

1. Implemented code.

2. Basic error handling.

3. Successful manual verification.

4. Relevant automated tests.

5. Updated documentation.

6. Externalized configuration when appropriate.

7. No hardcoded secrets.

8. Integration with the rest of the application.

9. Version-controlled source code.

10. Compliance with its acceptance criteria.

---

# 16. Project Success Criteria

The first two phases will be considered successful if the following complete workflow can be demonstrated:

```text
User
  ↓
Authentication
  ↓
Prediction Request
  ↓
Input Validation
  ↓
Preprocessing
  ↓
Model Router
  ↓
Selected Model
  ↓
Prediction
  ↓
Probability
  ↓
Database Record
  ↓
Response
```

The user must be able to switch between different models and use an automatic model-selection mode.

The system must also provide enough model metrics to explain why a specific model was selected as Champion.

Finally, the architecture must remain modular enough to support future datasets, predictive problems, sensors, and algorithms without requiring a complete redesign of the platform.