# Known Limitations

This document outlines known limitations of the CreditWise system.

These limitations are documented intentionally and transparently.

---

## Model Generalization

- The model is trained on a specific public dataset.
- Performance may not generalize across populations or regions.
- Retraining and revalidation are required for new contexts.

---

## Fairness Metrics

- Fairness metrics are observational only.
- No bias mitigation or enforcement is performed.
- Metrics may be sensitive to data imbalance and sampling bias.

---

## Explainability Constraints

- SHAP explanations assume feature independence approximations.
- LIME explanations are local and may be unstable.
- Explanations do not imply causality.

---

## Risk Band Interpretation

- Risk bands are heuristic groupings.
- Thresholds may not align with institutional risk policies.
- Bands are intended for review, not automated rejection.

---

## Data Limitations

- Dataset may contain historical bias.
- Sensitive attributes may be incomplete or noisy.
- Feature definitions are dataset-specific.

---

## Operational Scope

- Not optimized for low-latency production environments.
- Not designed for real-time decisioning.
- Intended for educational and demonstrative use.

---

## Regulatory Use

This system should not be used directly for regulated financial decision-making without additional validation, legal review, and governance controls.
