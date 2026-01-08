# Software Design Document (SDD) v0.05 Draft

## Cash Flow Forecasting Predictive Engine

---

## Document Control

| Attribute | Value |
|-----------|-------|
| **Document Version** | v0.05 Draft |
| **Date** | December 2025 |
| **Status** | Final Draft (Pending Architecture Review) |
| **Revision Authority** | Architecture Review Board |
| **Supersedes** | All prior SDD versions |
| **External Dependencies** | None |

---

## Table of Contents

1. Introduction & Revision Summary
2. Business Problem & Objectives
3. System Scope & Non-Goals
4. Unified Transaction Feed (UTF) Specification
5. Counterparty Reference Feed (CRF) Specification
6. Architecture Overview -- Layered Forecasting Stack
7. End-to-End Data Flow
8. Data Cleaning & Validation Layer
9. Consolidation, Transfer Netting & NECF Construction
10. Cash Flow Decomposition Layer
11. Outlier Detection & Treatment
12. Feature Engineering & Exogenous Events
13. Predictive Modeling Layer
14. Forecast Recomposition Logic
15. Explainability & Narrative Layer
16. Multi-Account, SME & Segmentation Strategy
17. Data History & Lookback Governance
18. Model Risk Management & Governance
19. Operational Considerations
20. Limitations & Known Trade-offs
21. Conclusion

**Appendices**

- Appendix A: Acceptance Test Plan (ATP) v0.05
- Appendix B: Model Selection Rules & Thresholds
- Appendix C: JSON Output Schemas
- Appendix D: Data Lineage & Audit Traceability
- Appendix E: Production Deployment Considerations

---

## 1. Introduction & Revision Summary

This document defines **SDD v0.05**, the authoritative design specification for the **Cash Flow Forecasting Predictive Engine**.

### Purpose of v0.05

Version **v0.05** represents a **major architectural evolution** from a PoC-centric design to a **production-aligned forecasting platform**, adopting **industry-standard layered forecasting practices** used by Tier-1 banks.

This document is **fully self-contained** and **does not require prior SDD versions** for interpretation.

### Key Architectural Shifts from Prior Versions

| Area | Prior State | v0.05 Definition |
|------|-------------|------------------|
| Forecast Target | Single-account net flow | **Consolidated Net External Cash Flow (NECF)** |
| Architecture | Monolithic SARIMA/SARIMAX | **Layered Hybrid Forecasting Stack** |
| Deterministic Knowledge | Implicit | **Explicit Layer 0 Rules** |
| Outlier Handling | On total net flow | **Residual-only treatment** |
| Model Stack | SARIMA / SARIMAX | **ETS, SARIMA, SARIMAX, optional ML** |
| Lookback | Fixed 24 months | **Dynamic: 24--60 months** |
| Explainability | Model-centric | **Decomposition-centric** |
| Governance | Implicit | **Explicit ATP, lineage, audit** |

---

## 2. Business Problem & Objectives

### 2.1 Business Problem

Customers ask a deceptively simple question:

> *"Based on my financial behavior, how much money will I realistically have in the coming months?"*

Answering this reliably is difficult because:

- Transaction histories contain **noise and anomalies**
- Customers have **multiple accounts**
- Internal transfers distort cash flow
- Known future events (loan endings, subscriptions) must be respected
- Forecasts must be **explainable and defensible**

### 2.2 Primary Objectives

1. Forecast **12-month future Net Cash Flow**
2. Support **multi-account consolidated customers**
3. Explicitly incorporate **known future contractual changes**
4. Deliver **high accuracy (WMAPE < 20%)**
5. Provide **full explainability and auditability**

---

## 3. System Scope & Non-Goals

### 3.1 In Scope

- Retail and SME customers
- Multi-account consolidation
- Monthly forecasting granularity
- Deterministic + statistical hybrid modeling
- Explainable outputs suitable for LLM consumption

### 3.2 Out of Scope (Explicit)

- Real-time transaction streaming
- Intraday forecasting
- Black-box deep learning (LSTM, transformers)
- Credit scoring or affordability decisions

---

## 4. Unified Transaction Feed (UTF) Specification

### 4.1 Purpose

The UTF is the **canonical, normalized source** of all historical customer transaction activity.

It serves as the **sole behavioral input** to the forecasting engine.

### 4.2 Core Fields (v0.05)

| Field | Type | Mandatory | Description |
|-------|------|-----------|-------------|
| **CustomerId** | String | Yes | Unique customer identifier (aggregation key) |
| **AccountId** | String | Yes | Source account |
| **TxID** | String | Yes | Unique transaction ID |
| **TxDate** | Date | Yes | Posting date |
| **Amount** | Float | Yes | Signed amount (+ credit, − debit) |
| **Currency** | String | Yes | ISO currency code |
| **Direction** | Enum | Yes | CREDIT / DEBIT |
| **Category** | String | Yes | Transaction category |
| **DescriptionRaw** | String | No | Raw description |
| **TransferLinkID** | String | No | Links internal transfer pairs |
| **IsRecurringFlag** | Boolean | Yes | Recurrence indicator |
| **IsVariableAmount** | Boolean | Yes | Amount variability |
| **CounterpartyKey** | String | Conditional | CRF join key |
| **RecurrenceStartDate** | Date | No | UTF-level start |
| **RecurrenceEndDate** | Date | No | UTF-level override |

### 4.3 UTF Constraints

- Mandatory fields must be present
- Amount must be numeric
- Currency must be normalized
- Duplicate TxIDs per account are invalid

---

## 5. Counterparty Reference Feed (CRF) Specification

### 5.1 Purpose

CRF provides **authoritative contractual truth** that cannot be inferred reliably from transactions alone.

### 5.2 CRF Responsibilities

- Define **true recurrence end dates**
- Define **contractual amounts**
- Override UTF ambiguity
- Provide future event certainty

### 5.3 Precedence Rules (Mandatory)

| Attribute | Precedence |
|-----------|------------|
| End Date | UTF override → Loan → Card Installment → Mandate → CRF generic |
| Amount | Loan → Card → Mandate → Historical median |
| Variability | CRF overrides UTF |

---

## 6. Architecture Overview -- Layered Forecasting Stack

### 6.1 Architectural Principle

**Deterministic knowledge must be applied before statistical learning.**

### 6.2 Layered Architecture

| Layer | Name | Purpose |
|-------|------|---------|
| **Layer 0** | Deterministic Rules | Transfer netting, recurrence, future deltas |
| **Layer 1** | Statistical Baselines | ETS, SARIMA, SARIMAX |
| **Layer 2** | ML Residuals (Optional) | Ridge / ElasticNet |
| **Layer 3** | Recomposition & Explainability | Final forecast assembly |

### 6.3 Why This Architecture Is Industry Standard

- Used by Tier-1 retail banks
- Separates **what is known** from **what must be learned**
- Improves stability, accuracy, and explainability
- Supports regulatory scrutiny

---

## 7. End-to-End Data Flow

1. UTF ingestion (multi-account)
2. Data cleaning & validation
3. UTF--CRF enrichment
4. **Transfer detection & netting**
5. **NECF construction**
6. **Cash flow decomposition**
7. Outlier treatment (residual only)
8. Exogenous delta preparation
9. Model training & selection
10. Forecast recomposition
11. Explainability package generation

---

## 8. Data Cleaning & Validation Layer

### 8.1 Objectives

- Ensure data integrity
- Prevent leakage into downstream logic
- Enable deterministic reconciliation

### 8.2 Validation Rules

- Reject missing CustomerId, AccountId, TxDate, Amount
- Normalize currencies
- Enforce valid dates
- Deduplicate using composite key

### 8.3 Governance Note

Cleaning **must occur before transfer netting** to avoid false mismatches.

---

## 9. Consolidation, Transfer Netting & NECF Construction

### 9.1 Problem Statement

In production environments, the majority of customers operate **multiple accounts** (e.g., checking, savings, joint, business). Naively aggregating transaction amounts leads to **double counting**, especially for **internal transfers**.

### 9.2 Transfer Detection

#### 9.2.1 Transfer Definition

An **internal transfer** is defined as a mirrored transaction pair satisfying all of the following:

- Same CustomerId
- Same absolute Amount
- Opposite Direction (DEBIT vs CREDIT)
- Occurring within a configurable time tolerance (default: ±2 days)
- Linked explicitly via TransferLinkID *or* inferred heuristically

#### 9.2.2 Transfer Matching Rules

Priority order:

1. Explicit TransferLinkID
2. Amount + Date proximity + Account mismatch
3. Category-based heuristics (e.g., TRANSFER_IN / TRANSFER_OUT)

### 9.3 Transfer Netting Logic

Once identified, both sides of an internal transfer are **fully excluded** from forecasting inputs.

**Rationale:** Internal transfers do not represent external cash entering or leaving the customer ecosystem.

### 9.4 Net External Cash Flow (NECF)

#### 9.4.1 Definition

$$
\textbf{NECF}_{m} = \sum \text{Amount}_{\text{external transactions in month } m}
$$

Where:

- External transactions exclude internal transfers
- Aggregation occurs **after netting**

#### 9.4.2 Aggregation Keys

- CustomerId
- MonthKey (YYYY-MM)

#### 9.4.3 Output

A clean, consolidated **monthly NECF time series**, forming the **foundation** for all downstream modeling.

---

## 10. Cash Flow Decomposition Layer

### 10.1 Objective

Separate **predictable cash flows** from **uncertain behavior**, improving both accuracy and explainability.

### 10.2 Decomposition Components

For each customer-month:

$$
\textbf{NECF} = \textbf{Deterministic Base} + \textbf{Residual}
$$

#### 10.2.1 Deterministic Base

Includes:

- Salary
- Contractual loans
- Installment plans
- Standing orders
- Direct debits with known schedules

Derived using:

- UTF recurrence flags
- CRF authoritative end dates and amounts
- Precedence rules defined in Section 5

#### 10.2.2 Residual Series

Represents:

- Discretionary spending
- Variable utilities
- One-off events
- Behavioral volatility

This is the **only component forecasted statistically**.

### 10.3 Integrity Constraint (Mandatory)

For every historical month:

$$
|\text{NECF} - (\text{Deterministic Base} + \text{Residual})| < \epsilon
$$

Where ε is floating-point tolerance.

---

## 11. Outlier Detection & Treatment

### 11.1 Design Principle

**Outliers are behavioral anomalies --- not contractual truths.**

### 11.2 Scope of Application

Outlier detection is applied **only to the Residual Series**, never to:

- Deterministic Base
- Known future deltas
- Contractual flows

### 11.3 Supported Methods

| Method | Usage |
|--------|-------|
| IQR | Default baseline |
| Z-score | Secondary |
| Isolation Forest | Optional (high-volume customers) |

### 11.4 Treatment Strategy

- Flag outliers
- Replace with rolling median or capped value
- Preserve original values for audit traceability

---

## 12. Feature Engineering & Exogenous Events

### 12.1 Feature Responsibilities

- Construct modeling target (Residual Series)
- Generate time-based features
- Create exogenous future delta vectors

### 12.2 Time-Series Features

- Lagged residuals (t−1, t−2, t−12)
- Month-of-year encoding
- Trend indicators

### 12.3 KnownFutureFlow_Delta Vector

#### 12.3.1 Definition

A deterministic vector representing **known future changes** in cash flow, derived from CRF events.

Examples:

- Loan maturity (+800)
- Installment plan end (+120)
- Insurance mandate termination (+80)

#### 12.3.2 Dual Usage (Critical)

1. **Model Input:** Exogenous regressor for SARIMAX
2. **Forecast Assembly:** Explicitly added during recomposition

This guarantees **zero information loss**, even when SARIMA is selected.

---

## 13. Predictive Modeling Layer

### 13.1 Modeling Philosophy

Prefer **simple, auditable models** unless complexity delivers material accuracy gains.

### 13.2 Supported Models

**Layer 1: Statistical Baselines**

| Model | Purpose |
|-------|---------|
| ETS | Robust baseline |
| SARIMA | Seasonal behavioral patterns |
| SARIMAX | Seasonal + known future events |

**Layer 2: Optional ML Models**

| Model | Usage Constraint |
|-------|------------------|
| Ridge Regression | High volatility residuals |
| ElasticNet | Correlated residual features |

**Explicitly excluded:** LSTM, Transformers (explainability & governance risk)

### 13.3 Training Target

- **Residual Series (cleaned)**

### 13.4 Evaluation Metric

$$
\textbf{WMAPE} = \frac{\sum |y - \hat{y}|}{\sum |y|}
$$

### 13.5 Model Selection Logic

- Lowest WMAPE wins
- Tie-breaker: simpler model
- Override allowed for explainability if within tolerance (≤0.5 pp)

---

## 14. Forecast Recomposition Logic

### 14.1 Final Forecast Equation

$$
\textbf{Forecast Total} = \textbf{Forecast Residual} + \textbf{Deterministic Base} + \textbf{Known Future Adjustments}
$$

### 14.2 Guarantees

- Contractual changes always reflected
- Forecast traceable per component
- Model error isolated to residual only

### 14.3 Output Granularity

- Monthly
- 12-month horizon
- Confidence intervals (optional)

---

## 15. Explainability & Narrative Generation Layer

### 15.1 Purpose

The Explainability Layer ensures that **every forecast is interpretable, auditable, and customer-safe**, enabling:

- Regulatory compliance
- Internal audit validation
- Customer-facing narrative generation (LLM-ready)

This layer converts raw numerical forecasts into **structured explanations** and **natural-language insights**.

### 15.2 Explainability Design Principles

1. **No Black Boxes**  
   Every forecast must be explainable as the sum of deterministic and statistical components.

2. **Causality Over Correlation**  
   Known future events (e.g., loan endings) must be explicitly cited as causes.

3. **LLM Safety**  
   All narratives must be grounded in structured JSON; no free inference is permitted.

### 15.3 Explainability JSON Contract (v0.05)

The system produces a single authoritative explainability payload:

**File:** `PoC_Forecast_Summary_v005.json`

#### 15.3.1 Mandatory Top-Level Fields

| Field | Description |
|-------|-------------|
| model_selected | Final winning model |
| model_candidates | All evaluated models with WMAPE |
| wmape_winner | Final accuracy score |
| meets_threshold | Boolean acceptance flag |
| forecast_period | Start, end, horizon |
| confidence_level | High / Medium / Low |

#### 15.3.2 Decomposition Metadata

```json
"cashflow_decomposition_summary": {
  "avg_necf": 472.3,
  "avg_deterministic_base": 420.0,
  "avg_residual": 52.3
}
```

#### 15.3.3 Transfer Netting Summary

```json
"transfer_netting_summary": {
  "num_transfers_removed": 18,
  "total_volume_removed": 9200.0
}
```

#### 15.3.4 Exogenous Event Traceability

Each future event must include:

| Field | Meaning |
|-------|---------|
| counterparty_display_name | Human-readable |
| end_month_key | YYYY-MM |
| delta_value | Monthly impact |
| applied_in_model | true/false |

### 15.4 Narrative Generation Contract (LLM Layer)

The forecasting engine **does not generate prose**.  
It produces **LLM-ready structured facts**.

The LLM layer must:

1. Use only provided fields
2. Cite causal events
3. Never invent amounts or dates
4. Respect confidence flags

---

## 16. Multi-Segment Support (Retail, SME, Corporate)

### 16.1 Segment Classification

Customers are dynamically classified based on:

| Feature | Usage |
|---------|-------|
| Account count | Complexity |
| Transaction volume | Volatility |
| Recurrence density | Predictability |
| Seasonality strength | Model choice |

### 16.2 Segment-Specific Behavior

| Segment | Modeling Notes |
|---------|----------------|
| **Retail** | Strong recurrence, SARIMAX preferred |
| **SME** | Higher volatility, ML residual optional |
| **Corporate** | Deterministic dominant, low residual |

### 16.3 Consolidated Accounts (Mandatory)

All segments must support:

- Multiple accounts
- Multiple currencies (normalized pre-model)
- Mixed transaction types

---

## 17. Model Governance & Risk Controls

### 17.1 Model Approval Policy

Only models explicitly listed in Section 13 are permitted.

**Disallowed without board approval:**

- Deep learning (LSTM, Transformers)
- Black-box AutoML

### 17.2 Drift Monitoring

Monitored metrics:

- Rolling WMAPE
- Residual variance
- Forecast bias

Triggers:

- Automatic retraining
- Confidence downgrade
- Alert to Risk & Analytics

### 17.3 Auditability Requirements

The system must retain:

- Raw UTF snapshot
- CRF snapshot
- Feature tables
- Model parameters
- Explainability JSON

Retention: **≥ 24 months**

---

## 18. Production Deployment Architecture

### 18.1 Execution Modes

| Mode | Use Case |
|------|----------|
| Batch | Monthly customer refresh |
| On-demand | Customer request |
| Event-driven | Major contract changes |

### 18.2 Stateless Design

Each forecast run is:

- Deterministic
- Reproducible
- Isolated

### 18.3 Performance Targets

| Metric | Target |
|--------|--------|
| Single customer forecast | < 300 ms |
| Batch 1M customers | < 2 hours |

---

## 19. Security & Privacy

- PII minimized in modeling layer
- CustomerId tokenized where possible
- Full GDPR audit trail supported
- No raw transaction text exposed to LLMs

---

## Appendix A --- Full Acceptance Test Plan (ATP) v0.05

### Summary

| Category | Test Cases |
|----------|------------|
| Data Integrity | TC-01 → TC-05 |
| Recurrence Logic | TC-06 → TC-08 |
| Modeling Accuracy | TC-09 |
| Explainability | TC-10 |
| Consolidation | TC-11 → TC-13 |
| Model Benchmarking | TC-14 |
| Explainability Completeness | TC-15 |

All tests must PASS for production promotion.

---

## Appendix B --- Data Lineage & Traceability

End-to-end lineage:

```
UTF → Cleaning → Enrichment → Netting → Decomposition → Residual → Model → Recomposition → Explainability JSON → LLM Narrative
```

---

## Appendix C --- JSON Schema Definitions

- ForecastSummarySchema_v005.json
- ModelCandidateSchema.json
- ExogenousEventSchema.json

(All schemas versioned and immutable.)

---

## Appendix D --- Known Limitations

- Short history (<24 months) flagged low confidence
- High-volatility SMEs may require manual review
- Currency FX shocks handled upstream

---

## Appendix E --- Change Log

| Version | Key Change |
|---------|------------|
| v0.03 | Single account, SARIMA |
| v0.04 | CRF + SARIMAX |
| **v0.05** | **Layered architecture, consolidation, production governance** |

---

## Document Status

**SDD v0.05 Draft --- COMPLETE**

This document is **fully standalone**, production-aligned, and compliant with industry-standard layered forecasting architectures.