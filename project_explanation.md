# FRANQ-EXT Project Explanation
*A Plain-English Guide to Understand the Project Before Implementation*

---

# 1. What is this project?

This project is **not a chatbot**.

It is **not another RAG application**.

Instead, it is a **research project** whose goal is to make **Retrieval-Augmented Generation (RAG)** systems more trustworthy.

The project can be summarized in one sentence:

> **Given a RAG answer, automatically detect incorrect facts, intelligently verify them, correct only the wrong facts, and finally ensure that the correction itself did not introduce new mistakes.**

Think of this project as building an **AI Fact Checker + AI Self-Correction System**.

---

# 2. What problem does it solve?

Suppose a user asks:

> Where was Albert Einstein born?

A normal RAG system retrieves relevant documents and generates:

```
Einstein was born in 1879 in Munich.
```

Looks reasonable.

But actually,

| Fact | Correct? |
|------|----------|
| Birth Year = 1879 | ✅ Correct |
| Birth City = Munich | ❌ Wrong |
| Correct Birth City | Ulm |

Only one fact is incorrect.

Current RAG systems usually:

- Generate an answer
- Sometimes detect that something looks wrong

But they rarely:

- Identify exactly **which fact** is wrong
- Correct **only that fact**
- Verify that the correction didn't accidentally break another correct fact

This project solves that problem.

---

# 3. What is FRANQ?

The project extends an existing research paper called **FRANQ**.

FRANQ already tries to improve RAG by checking:

- **Faithfulness**
  - Does the answer match the retrieved documents?

- **Factuality**
  - Is the answer actually true?

FRANQ workflow:

```
Question
      ↓
Retrieve Documents
      ↓
LLM Generates Answer
      ↓
Verification
      ↓
Report
```

It mainly **detects** problems.

It does **not repair** them.

---

# 4. What are the problems with FRANQ?

According to the project explainer, FRANQ has three weaknesses.

---

## Problem 1 — Facts are checked independently

Example:

```
Einstein

Birth Year = 1879

Birth City = Munich
```

FRANQ checks:

```
Fact 1

Einstein born in 1879

Fact 2

Einstein born in Munich
```

as two completely separate facts.

But both belong to the same person.

If the entity itself is suspicious,

all connected facts should also become suspicious.

FRANQ cannot understand these relationships.

---

## Problem 2 — Fixed Decision Rule

FRANQ always uses a hardcoded IF-ELSE rule.

Example:

```python
if faithfulness_score < threshold:
    use_method_A()
else:
    use_method_B()
```

Every question uses the same rule.

Real-world questions are different.

Different situations require different verification strategies.

---

## Problem 3 — No Correction

FRANQ says:

```
This fact is probably wrong.
```

Then stops.

It never:

- fixes the answer
- verifies the correction
- checks if the correction introduced a new mistake

---

# 5. What is your contribution?

This project extends FRANQ using **three major research contributions (Pillars).**

---

# Pillar 1

## Dependency-Aware Entity–Attribute Graph

Instead of treating facts independently,

the system builds a graph.

Instead of

```
Fact 1

Fact 2

Fact 3
```

it creates

```
Einstein

├── Birth Year
├── Birth City
├── Occupation
├── Nationality
```

Now all related facts are connected.

If one fact becomes suspicious,

related facts can also be verified.

This is the first novelty.

---

# Pillar 2

## Learned Calibrated Router

Instead of using

```python
if score < threshold
```

the project trains a machine learning model.

Input features include:

- Faithfulness Score
- Retrieval Confidence
- Entity Type
- Dependency Strength
- Semantic Entropy
- Parametric Knowledge Uncertainty

↓

The router decides:

```
Does this fact require deeper verification?

YES

or

NO
```

Instead of a hardcoded rule,

the decision becomes adaptive.

---

# Pillar 3

## Targeted Self Correction

Suppose the generated answer is

```
Einstein was born in Munich.
```

Instead of regenerating the entire answer,

the system searches specifically for

```
Einstein birth city
```

retrieves better evidence,

and replaces only

```
Munich

↓

Ulm
```

Everything else remains unchanged.

---

# 6. New Metric — Correction Regret

Suppose before correction:

| Fact | Status |
|------|--------|
| Birth Year | ✅ Correct |
| Birth City | ❌ Wrong |

After correction:

| Fact | Status |
|------|--------|
| Birth Year | ❌ Wrong |
| Birth City | ✅ Correct |

The system fixed one thing,

but accidentally broke another.

To detect this,

the project introduces a new metric:

## Correction Regret

```
Correction Regret

=

Previously Correct Facts
that became Wrong

--------------------------------

Total Previously Correct Facts
```

Smaller is better.

This is one of the major research contributions.

---

# 7. Complete Pipeline

Entire workflow:

```
User Question
        ↓
Retriever
        ↓
LLM
        ↓
Generated Answer
        ↓
Fact Extraction
        ↓
Entity Graph
        ↓
Faithfulness Score
        ↓
Uncertainty Estimation
        ↓
Learned Router
        ↓
Need Correction?
        ↓
Targeted Retrieval
        ↓
Replace Wrong Fact
        ↓
Reverification
        ↓
Correction Regret
        ↓
Final Correct Answer
```

---

# 8. What modules will you implement?

## Module 1

### Fact Extractor

Input:

```
Einstein was born in 1879 in Munich.
```

Output:

```
Entity

Einstein

Attribute

Birth City

Value

Munich
```

---

## Module 2

### Entity Graph

Creates

```
Einstein

↓

Birth Year

↓

Birth City

↓

Occupation
```

---

## Module 3

### Dense Retriever

Retrieves relevant evidence using semantic search.

Likely technologies:

- Sentence Transformers
- FAISS

---

## Module 4

### AlignScore

Measures:

> Does the retrieved evidence actually support this fact?

---

## Module 5

### Semantic Entropy

Ask the LLM multiple times.

If answers vary significantly,

the model is uncertain.

---

## Module 6

### Parametric Knowledge Uncertainty

Measures

> Does the model itself know this information confidently?

---

## Module 7

### Learned Router

Combines:

- AlignScore
- Retrieval Confidence
- Semantic Entropy
- Parametric UQ
- Graph Features

↓

Outputs:

```
Need verification?

YES / NO
```

---

## Module 8

### Corrector

Only edits

incorrect facts.

Does not regenerate the whole answer.

---

## Module 9

### Reverification

Checks:

```
Did the correction break anything else?
```

Calculates

Correction Regret.

---

## Module 10

### Experiments

Compare

```
FRANQ

↓

FRANQ + Entity Graph

↓

FRANQ + Graph + Router

↓

Full Proposed System
```

Generate:

- CSV tables
- Figures
- Ablation study
- Final results

---

# 9. Technologies You Will Learn

This project introduces several advanced LLM research topics.

Technologies include:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (Llama 3.2)
- HuggingFace Transformers
- Sentence Transformers
- FAISS
- AlignScore
- Semantic Entropy
- Uncertainty Quantification
- Calibration
- Scikit-learn
- PyTorch

Evaluation Metrics:

- Faithfulness
- Factual Accuracy
- AUROC
- PRR
- Calibration Error (ECE)
- Correction Success Rate
- Correction Regret

---

# 10. Comparison with MedMaternityChain

| MedMaternityChain | FRANQ-EXT |
|-------------------|------------|
| AI + Blockchain | Advanced LLM Research |
| Federated Learning | RAG Verification |
| Streamlit Dashboard | Research Pipeline |
| Healthcare Prediction | Hallucination Detection |
| Explainable AI | Fact Verification |
| Product Demo | Research Prototype |

---

# 11. Overall Understanding

This project is **more research-oriented** than MedMaternityChain.

Instead of building an end-user healthcare application,

you are building an **AI research framework** that improves the reliability of Retrieval-Augmented Generation systems.

The focus is not on generating answers,

but on making sure that generated answers are:

- faithful to retrieved evidence
- factually correct
- intelligently verified
- automatically corrected
- safely reverified

This makes the project highly relevant to modern LLM research and suitable for publication-oriented academic work.

---

# Final Summary

In one sentence:

> **FRANQ-EXT is a next-generation RAG verification framework that extends FRANQ by introducing dependency-aware fact verification, an intelligent uncertainty-based routing mechanism, targeted self-correction, and a novel correction-regret metric to ensure that fixing one fact never damages another.**