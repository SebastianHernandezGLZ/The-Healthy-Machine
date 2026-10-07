# The Healthy Machine

## 1. Overview

**The Healthy Machine** is an Artificial Intelligence-based predictive maintenance platform whose purpose is to analyze data from machinery, sensors, and industrial systems to detect abnormal behavior, estimate failure risks, and support maintenance-related decision-making.

The project is designed as an extensible and modular platform capable of working with multiple Machine Learning models. Instead of relying on a single algorithm, the system will allow different predictive models to be trained, evaluated, registered, compared, and selected to perform inferences.

The platform will be developed progressively with the goal of integrating knowledge from different areas of Artificial Intelligence Engineering, including Machine Learning, software engineering, databases, cybersecurity, embedded systems, API development, MLOps, monitoring, software testing, and application deployment.

The project does not seek only to obtain a model with good metrics, but to build a complete system around the lifecycle of Artificial Intelligence models.

---

## 2. Problem

Industrial machines and equipment may experience gradual degradation before a failure occurs. Variables such as temperature, rotational speed, torque, wear, vibration, current, or pressure may contain patterns that make it possible to identify abnormal behavior before a breakdown becomes critical.

Traditional maintenance is usually performed through two main approaches:

**Corrective maintenance:** action is taken after the failure occurs.

**Preventive maintenance:** maintenance is performed according to predetermined intervals, even when the machine may be in good condition.

Predictive maintenance proposes using historical data and data obtained through sensors to estimate the condition of a machine and anticipate possible failures.

The platform developed in this project will seek to apply different Artificial Intelligence algorithms to analyze this data and generate useful predictions.

---

## 3. General objective

Design and develop a modular Artificial Intelligence-based predictive maintenance platform that makes it possible to train, compare, manage, and use multiple predictive models to determine machinery failure risk from operational data.

---

## 4. Specific objectives

The main objectives of the project are:

- Implement a reproducible pipeline for data loading, validation, cleaning, and transformation.
- Develop different Machine Learning models to solve the same predictive problem.
- Establish a common interface that allows models to be exchanged without modifying the rest of the system.
- Evaluate models using appropriate metrics for imbalanced classification problems.
- Implement a mechanism to manually or automatically select the model used during inference.
- Register versions, metrics, and configurations of trained models.
- Expose inference capabilities through an API.
- Store the history of predictions made by the system.
- Implement authentication, authorization, and basic security controls.
- Record relevant events through auditing mechanisms.
- Implement unit and integration tests.
- Incorporate experiment tracking and model lifecycle management tools.
- Create a visual interface that allows interaction with the platform.
- Prepare the system for future integrations with IoT devices and physical sensors.

---

## 5. Initial scope

During the first version of the project, the platform will focus on a **binary classification** problem.

The system will receive information related to a machine's operating conditions and must determine whether or not there is a risk of failure.

Conceptually:

```text
Machine data
     ↓
Data validation
     ↓
Preprocessing
     ↓
Model selection
     ↓
Machine Learning model
     ↓
Prediction
     ↓
Failure / No Failure
```

The platform will allow different models to be selected to perform the same inference and compare their behavior.

Initially, models such as the following will be used:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

New algorithms may later be incorporated without significantly modifying the existing architecture.

---

## 6. Initial dataset

The first dataset used will be **AI4I 2020 Predictive Maintenance Dataset**.

This dataset contains information related to machinery operating conditions and was designed for experimenting with predictive maintenance problems.

Among the available variables are:

- Product type.
- Air temperature.
- Process temperature.
- Rotational speed.
- Torque.
- Tool wear.
- Machine failure indicator.
- Different failure categories.

The initial objective will be to use the available operational variables to predict the variable:

```text
Machine failure
```

where:

```text
0 = No failure
1 = Failure
```

This dataset will make it possible to develop the first functional version of the complete Machine Learning pipeline.

---

## 7. Future datasets

The project is designed to grow progressively and work with different types of predictive maintenance problems.

### NASA C-MAPSS

In future versions, the **NASA C-MAPSS** dataset will be used, which contains simulated data related to the progressive degradation of turbofan engines.

This dataset will make it possible to address problems involving:

- regression;
- time series;
- component degradation;
- Remaining Useful Life (RUL) estimation.

The objective will be to estimate approximately how much operating time a component may have left before reaching a failure condition.

---

### PRONOSTIA

The **PRONOSTIA** dataset, related to bearing degradation, may also be incorporated.

It will make it possible to work with vibration signals and apply techniques such as:

- signal analysis;
- feature extraction;
- FFT;
- classification models;
- neural networks;
- 1D CNN;
- anomaly detection.

---

### Battery data

In later stages, datasets related to battery degradation may be used to experiment with capacity prediction, state of health, and remaining useful life.

---

### Data obtained through physical sensors

One of the most important future extensions will be to generate a proprietary dataset using real hardware.

A possible physical system will include:

```text
Sensors
   ↓
ESP32
   ↓
Communication protocol
   ↓
Backend
   ↓
Predictive platform
```

Variables such as the following may be obtained:

- temperature;
- vibration;
- current;
- voltage;
- speed;
- RPM.

This will make it possible to connect knowledge of embedded systems, IoT, and Artificial Intelligence within the same platform.

---

## 8. Multiple-model strategy

A central feature of the project will be the ability to use multiple predictive models.

All models must follow a common interface.

Conceptually:

```text
BaseModel

├── LogisticRegressionModel
├── DecisionTreeModel
├── RandomForestModel
└── XGBoostModel
```

Each implementation must provide equivalent operations, such as:

```text
train()

predict()

predict_proba()

evaluate()

save()

load()
```

This will allow other system components to interact with any model without knowing the internal details of its implementation.

---

## 9. Model selection

During inference, the system may use different strategies to select the model.

### Manual selection

The user will be able to select a model directly.

For example:

```text
Logistic Regression
Random Forest
XGBoost
Decision Tree
```

---

### Automatic selection

There will also be an option:

```text
AUTO
```

In the first version, this option will use the model considered to be the best available model according to the criteria defined by the system.

In later versions, more advanced dynamic selection strategies may be developed.

---

## 10. Model evaluation

Models will not be evaluated only through accuracy.

Because machinery failures may represent a minority class, it will be necessary to use metrics capable of correctly measuring performance on imbalanced datasets.

The main metrics will be:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Characteristics related to system operation may also be evaluated, such as:

- inference time;
- resource consumption;
- model size;
- stability;
- model version.

The selection of the best model will not necessarily depend on a single metric.

---

## 11. Model Router

The platform will include a component called **Model Router**.

This component will be responsible for determining which model should receive an inference request.

Example:

```text
Prediction Request
       ↓
   Model Router
       ↓
 ┌─────┼──────────────┐
 ↓     ↓              ↓
RF   XGBoost       Logistic
 ↓     ↓              ↓
 └─────┴──────┬───────┘
              ↓
         Prediction
```

Initially, the router will select models using simple rules.

Later, it may use information such as:

- model performance;
- data type;
- required latency;
- version;
- availability;
- model status;
- resource consumption.

---

## 12. Model lifecycle management

The platform will seek to manage models as versioned and manageable components.

Each model may store information such as:

```text
Model name
Model version
Training date
Dataset version
Hyperparameters
Metrics
Status
```

It is also planned to implement statuses such as:

```text
Experimental
Candidate
Champion
Archived
```

This will make it possible to keep a main model in production while other models are evaluated.

---

## 13. Champion / Challenger

In later stages of the project, the **Champion / Challenger** concept will be incorporated.

The Champion model will be considered the main model used for inference.

Example:

```text
Random Forest v3
Status: Champion
```

Meanwhile, a new model may be registered as Challenger.

```text
XGBoost v5
Status: Challenger
```

The system will be able to compare both models using technical and Machine Learning metrics.

If the Challenger demonstrates better results under the established criteria, it may become the new Champion.

Conceptually:

```text
New Model
    ↓
Evaluation
    ↓
Candidate
    ↓
Challenger
    ↓
Comparison
    ↓
Promotion
    ↓
Champion
```

---

## 14. Software architecture

The project will follow a modular architecture where each component will have a specific responsibility.

In simplified form:

```text
                User
                  ↓
               Frontend
                  ↓
                 API
                  ↓
            Service Layer
                  ↓
           Inference Layer
                  ↓
             Model Router
                  ↓
        Machine Learning Models
                  ↓
               Result
```

Other components will be incorporated around this architecture:

```text
Database

Authentication

Authorization

Audit Logging

Experiment Tracking

Monitoring

Model Registry
```

The separation between components will make it possible to modify one part of the platform without unnecessarily affecting the others.

---

## 15. Software engineering principles

The project will be developed considering design principles that promote maintainability, scalability, and extensibility.

Concepts such as the following will be applied:

- Separation of Concerns.
- Single Responsibility Principle.
- Dependency Inversion.
- Interfaces and abstractions.
- Low coupling.
- High cohesion.
- External configuration.
- Structured error handling.
- Logging.
- Testing.

Design patterns may also be applied when there is a real need.

Among the planned patterns are:

### Strategy Pattern

It will make it possible to interchange different prediction algorithms.

### Factory Pattern

It will centralize model creation.

### Repository Pattern

It will separate business logic from database access.

Patterns will not be used solely for demonstration purposes; they must solve a real problem within the architecture.

---

## 16. API

The platform will expose its functionalities through an API initially developed with FastAPI.

Some planned endpoints are:

```text
GET /health

GET /models

GET /models/metrics

POST /predict

POST /models/select
```

Endpoints related to the following may later be incorporated:

```text
authentication

users

training

model management

audit logs

monitoring
```

The API will allow different interfaces or applications to consume the platform's capabilities without depending directly on the internal implementation of the models.

---

## 17. Persistence

The platform will use a database to store information related to the system.

Among the data that may be stored are:

- users;
- models;
- versions;
- metrics;
- predictions;
- experiments;
- audit events.

PostgreSQL is planned for the first complete version.

The prediction history will make it possible to later analyze the behavior of the models and the system.

---

## 18. MLOps

One of the goals of the project will be to progressively incorporate MLOps practices.

These practices will make it possible to manage not only code, but also experiments, models, configurations, and results.

It is planned to use tools such as MLflow to:

- log experiments;
- store hyperparameters;
- compare metrics;
- version models;
- manage artifacts;
- identify candidate models;
- maintain training traceability.

Conceptually:

```text
Dataset
   ↓
Training
   ↓
Evaluation
   ↓
Experiment Tracking
   ↓
Model Registry
   ↓
Candidate
   ↓
Production
```

---

## 19. Security

Security will be considered as part of the architecture and not as a functionality added only at the end.

Among the controls that may be incorporated are:

- authentication;
- authorization;
- JWT;
- roles;
- permissions;
- password hashing;
- input validation;
- rate limiting;
- environment variables;
- secret management;
- HTTPS;
- audit logging.

Initially, roles such as the following may exist:

```text
ADMIN

ML_ENGINEER

OPERATOR

VIEWER
```

Each role will have different permissions within the system.

A threat analysis using methodologies such as STRIDE may be performed later.

---

## 20. Testing

The project will incorporate testing from its earliest stages.

Tests will mainly be divided into:

### Unit Tests

They will validate functions and components in isolation.

### Integration Tests

They will verify that different components work correctly when used together.

### Model Tests

They will validate model-specific aspects such as prediction structure, probabilities, expected features, and minimum performance.

### Security Tests

They will verify restrictions related to authentication, authorization, and permissions.

---

## 21. Monitoring

In later versions, the platform will incorporate mechanisms to observe both application behavior and model performance.

Among the data that may be monitored are:

```text
Number of predictions

Inference latency

Errors

Model usage

Prediction distribution

Model performance

Data distribution
```

It is also planned to incorporate detection mechanisms for:

```text
Data Drift

Concept Drift
```

This will make it possible to identify when a model may need evaluation or retraining.

---

## 22. User interface

In the second phase of the project, a visual interface will be developed to allow interaction with the platform.

The interface may allow users to:

- enter machine data;
- select a model;
- use automatic selection;
- make a prediction;
- check the failure probability;
- visualize metrics;
- compare models;
- check the prediction history;
- observe model versions and statuses.

Conceptual example:

```text
Predictive Maintenance Platform

Machine Data
----------------------------

Air Temperature:      [   ]

Process Temperature:  [   ]

Rotational Speed:     [   ]

Torque:               [   ]

Tool Wear:            [   ]

Model:
[AUTO ▼]

[PREDICT]


Result

Failure probability: 84%

Prediction: FAILURE

Model used: Random Forest v3
```

---

## 23. Deployment

The different project components will be progressively prepared to run through containers.

Docker is planned for components such as:

```text
Backend

Frontend

PostgreSQL

MLflow
```

The goal will eventually be to allow the complete environment to be started through:

```text
docker compose up
```

This will make it easier to reproduce the project on different computers and prepare future deployments on cloud infrastructure.

---

## 24. Initial development phases

Initial development will be divided into two main phases.

### Phase 1 — Machine Learning Core

Approximate duration:

```text
1 month
```

Main objectives:

- repository setup;
- documentation;
- AI4I analysis;
- preprocessing;
- Logistic Regression;
- Decision Tree;
- Random Forest;
- XGBoost;
- evaluation;
- comparison;
- common model architecture;
- Model Factory;
- Model Manager;
- Model Router;
- FastAPI;
- basic testing.

Expected result:

```text
Dataset
   ↓
Preprocessing
   ↓
Multiple Models
   ↓
Model Router
   ↓
API
   ↓
Prediction
```

---

### Phase 2 — Integrated platform

Approximate duration:

```text
1 month
```

Main objectives:

- PostgreSQL;
- prediction history;
- experiment tracking;
- MLflow;
- model versions;
- Champion / Challenger;
- authentication;
- authorization;
- RBAC;
- audit logging;
- dashboard;
- Docker;
- integration testing;
- final documentation.

Expected result:

```text
Frontend
   ↓
FastAPI
   ↓
Authentication
   ↓
Services
   ↓
Model Router
   ↓
Multiple ML Models
   ↓
Prediction
   ↓
PostgreSQL

+

MLflow

+

Audit Logging

+

Model Management
```

---

## 25. Future extensions

The project may continue growing after the first two phases.

Possible extensions include:

- NASA C-MAPSS.
- Remaining Useful Life.
- Time series.
- LSTM.
- GRU.
- 1D CNN.
- Autoencoders.
- Anomaly Detection.
- Ensemble Learning.
- Dynamic Model Routing.
- Data Drift.
- Concept Drift.
- Retraining pipelines.
- ESP32.
- Physical sensors.
- MQTT.
- Edge Computing.
- Cloud deployment.
- CI/CD.
- Kubernetes.
- Microservices.
- Distributed inference.
- Model security.
- Adversarial Machine Learning.

These functionalities are not a mandatory part of the initial MVP, but the architecture must facilitate their future incorporation.

---

## 26. Expected project outcome

At the end of the first two phases, the platform must be capable of:

1. Loading and processing predictive maintenance data.

2. Training different Machine Learning models.

3. Evaluating each model through reproducible metrics.

4. Comparing models.

5. Saving trained models.

6. Versioning models.

7. Selecting different models for inference.

8. Automatically selecting a main model.

9. Receiving information through an API.

10. Making failure predictions.

11. Returning prediction probabilities.

12. Saving an inference history.

13. Managing users and basic permissions.

14. Recording relevant actions through auditing.

15. Consulting experiments and metrics.

16. Interacting with the system through a visual interface.

17. Running its main components through Docker.

---

## 27. Long-term vision

The final vision is to transform the project into a complete experimental platform for studying the lifecycle of Artificial Intelligence systems applied to predictive maintenance.

The system must allow new datasets, algorithms, sensors, architectures, and inference strategies to be progressively incorporated without completely redesigning the application.

The platform will serve as an integrative project where knowledge acquired across different areas of Artificial Intelligence Engineering can be applied.

The goal will not be only to build models capable of making predictions, but to develop the infrastructure necessary for those models to be trained, evaluated, deployed, monitored, protected, replaced, and used within a maintainable software system.

In this way, the project will function both as a technical learning platform and as a professional demonstration of knowledge in Artificial Intelligence, Machine Learning, software engineering, MLOps, cybersecurity, databases, embedded systems, and intelligent systems development.
