# Engineering Decisions

This document records key design decisions made during the development of CreditWise, along with their rationale and tradeoffs.

The goal is to make assumptions and constraints explicit.

---

## Model Choice: CatBoost

**Decision**  
Use CatBoost for credit risk prediction.

**Rationale**
- Native handling of categorical features
- Strong and stable performance on tabular data
- Reduced preprocessing complexity
- Well-supported SHAP integration

**Tradeoffs**
- Less flexible than custom neural architectures
- Model size larger than linear alternatives

---

## Explainability Strategy: SHAP + LIME

**Decision**  
Support both SHAP and LIME for local explainability.

**Rationale**
- SHAP provides model-consistent explanations
- LIME offers model-agnostic local approximations
- Enables comparison between explanation methods

**Tradeoffs**
- LIME explanations may be unstable
- SHAP computation can be expensive for batch runs

Explainability is treated as a **diagnostic layer**, not a control mechanism.

---

## Fairness Monitoring (Not Enforcement)

**Decision**  
Compute fairness metrics for monitoring only.

**Rationale**
- Automated fairness enforcement can introduce unintended harm
- Regulatory and ethical interpretations vary
- Human review is required in high-stakes domains

**Tradeoffs**
- Does not correct bias automatically
- Requires downstream governance processes

This decision mirrors real-world enterprise practices.

---

## Batch-Oriented Workflows

**Decision**  
Support batch inference and offline reporting.

**Rationale**
- Reflects enterprise audit and review workflows
- Enables asynchronous governance
- Avoids real-time decision coupling

**Tradeoffs**
- Higher latency
- Increased system complexity

---

## Separation of System Components

**Decision**  
Explicitly separate prediction, explainability, fairness, and reporting modules.

**Rationale**
- Prevents hidden dependencies
- Enables independent inspection and testing
- Improves maintainability

**Tradeoffs**
- More boilerplate code
- Additional orchestration required

---

## UI as an Orchestration Layer

**Decision**  
Limit the Gradio UI to orchestration and visualization.

**Rationale**
- Keeps business logic testable and reusable
- Prevents UI-driven coupling
- Supports future interface changes

---

## Documentation of Limitations

**Decision**  
Explicitly document known limitations.

**Rationale**
- Encourages responsible use
- Avoids overclaiming system capability
- Supports ethical and regulatory transparency

This is a deliberate design choice, not an omission.
