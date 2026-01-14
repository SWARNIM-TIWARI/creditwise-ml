# System Architecture

This document describes the architecture of the CreditWise explainable credit risk system, focusing on data flow, component responsibilities, and separation of concerns.

The system is designed to prioritize **auditability, explainability, and governance** over raw predictive performance.

---

## High-Level Overview

CreditWise is an end-to-end tabular ML system that supports:

- Single-instance inference with local explanations
- Batch inference with governance-ready reporting
- Observational fairness analysis
- Auto-generated model documentation

The architecture intentionally separates **model logic**, **explainability**, **fairness**, and **presentation** layers to avoid hidden coupling and to support inspection.

---

## Data Flow

Raw Dataset (CSV)
↓
Data Preprocessing
↓
Feature-Ready Dataset
↓
CatBoost Model
↓
Prediction Probabilities
↓
Risk Band Mapping
├── SHAP & LIME Explanations
├── Fairness Metrics
└── Batch PDF Reports + Model Card

yaml
Copy code

---

## Core Components

### 1. Data Preparation (`data_prep.py`)
Responsible for:
- Cleaning raw tabular data
- Handling missing values
- Encoding categorical features
- Producing feature-consistent train/test/validation splits

All preprocessing logic is centralized to ensure consistency between training and inference.

---

### 2. Model Training (`train_model.py`)
- Trains a CatBoost classifier on tabular credit data
- Persists the trained model and associated metadata
- Captures feature ordering and categorical column indices

Training artifacts are treated as **derived outputs**, not source code.

---

### 3. Prediction & Risk Scoring (`predict.py`)
- Loads trained model and metadata
- Produces default probability scores
- Maps probabilities to interpretable risk bands

Risk bands are designed for human review rather than automated decision enforcement.

---

### 4. Explainability Layer (`explainability.py`)
Provides **local explanations** for individual predictions using:
- SHAP (TreeExplainer) for model-consistent explanations
- LIME for approximate local explanations

Explainability is executed **post-prediction** and does not influence model outputs.

---

### 5. Fairness Monitoring (`fairness.py`)
- Computes group-level outcome statistics across sensitive attributes
- Surfaces risk score deltas between groups
- Produces visualizations for monitoring purposes

Fairness metrics are observational only and are not used to alter predictions.

---

### 6. Reporting & Governance (`report.py`, `model_card.py`)
- Generates batch PDF reports for offline review
- Produces a model card summarizing objectives, metrics, and limitations
- Supports governance and audit workflows

---

### 7. Application Layer (`app.py`)
- Gradio-based UI for interaction and visualization
- Orchestrates calls to core pipeline components
- Contains no model logic

---

## Design Principles

- Explicit separation of concerns
- Transparency over automation
- Human-in-the-loop review support
- Failure awareness over silent optimization

---

## Intended Use

This system is intended for:
- Educational purposes
- ML system design demonstrations
- Explainability and governance research

It is **not intended for live credit decisioning**.
