# MovieLens 100K Rating Classification (No Metadata)

## Overview

This project implements a complete **multi-class rating classification pipeline** on the MovieLens 100K dataset using **ONLY user IDs and ratings** (as required). No user metadata (age, gender, occupation) or movie metadata (genres, titles, tags) is used anywhere in the pipeline.

The task is formulated as a **5-class classification problem** where ratings ∈ {1,2,3,4,5}.

The implementation includes:

* Baseline bias model
* Matrix Factorization via Truncated SVD
* Classical ML classifiers
* Gradient boosting models
* Neural Collaborative Filtering (NeuralCF)
* Siamese-style Collaborative Filtering network
* Cross-validation on official 5 predefined folds

---

## Dataset

MovieLens 100K

* 100,000 ratings
* 943 users
* 1682 movies
* Ratings: 1–5

Evaluation is performed using the official 5-fold split:

* u1.base / u1.test
* u2.base / u2.test
* u3.base / u3.test
* u4.base / u4.test
* u5.base / u5.test

---

## Problem Formulation

We model rating prediction as:

Multi-class classification:

[ f(user, item) → {1,2,3,4,5} ]

Metrics used:

* **Accuracy**
* **Macro F1-score**

Only these two metrics are considered for final evaluation.

---

## Feature Engineering (No Metadata Used)

All features are derived strictly from:

* user_id
* item_id
* rating (training fold only)

### 1 Baseline Bias Model

We compute:

[
\hat{r}_{ui} = \mu + b_u + b_i
]

Where:

* μ = global mean rating
* b_u = user bias
* b_i = item bias

This acts as a strong classical collaborative filtering baseline.

---

### 2 User & Item Statistics

Computed from training fold only:

* User mean rating
* User rating count
* Item mean rating
* Item rating count

These capture popularity and rating tendencies.

---

### 3 Matrix Factorization (Truncated SVD)

We build a sparse user-item rating matrix and apply:

TruncatedSVD (latent_dim = 60)

This produces:

* User latent vectors (60-dim)
* Item latent vectors (60-dim)

Final feature vector per interaction:

[ μ, b_u, b_i,
u_mean, u_count,
i_mean, i_count,
user_latent_1 … user_latent_60,
item_latent_1 … item_latent_60 ]

Total ≈ 127 numerical features.

All numeric features are standardized.

---

## Models Implemented

### 1 Baseline Bias Model

Direct rounding of μ + b_u + b_i.

---

### 2 Linear Models

* Logistic Regression (Multinomial)
* SGD (log-loss)
* Linear SVC

These serve as strong linear baselines on latent features.

---

### 3 Tree-Based Models

* Random Forest
* Extra Trees
* XGBoost
* LightGBM
* CatBoost

These capture nonlinear interactions between user and item latent factors.

Hyperparameter tuning is done conservatively using Optuna on Fold 1 only.

---

### 4 MLP (Sklearn)

Feedforward neural network on engineered numeric features.

Architecture:

* 512 → 256 → Softmax(5)

---

### 5 Neural Collaborative Filtering (NeuralCF)

Deep model using embeddings:

* User embedding (64-dim)
* Item embedding (64-dim)
* Concatenation
* Dense(256) → Dense(128) → Softmax(5)

This learns nonlinear user-item interactions directly.

---

### 6 Siamese Collaborative Filtering Network

Siamese-style architecture:

* Separate embeddings for user and item
* Shared dense tower (shared weights)
* Absolute difference + concatenation
* Final classification head

This enforces symmetric representation learning between users and items.

---
### 7 GCN council 

* Implemented an ensemble of multiple independently trained GCMC models.  
* Each model uses different random initialization and stochastic dropout, leading to diverse learned representations.  
* Final prediction is obtained by averaging the logits from all models in the council.  
* This reduces variance and improves generalization compared to a single model.  
* The GNN Council achieved the best overall accuracy among all implemented approaches.

## Cross-Validation Results
### Per-Fold Results (Accuracy / Macro-F1)

| Model        | Fold 1              | Fold 2              | Fold 3              | Fold 4              | Fold 5              |
| ------------ | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Baseline     | 0.4017 / 0.2702     | 0.4103 / 0.2752     | 0.4077 / 0.2658     | 0.4106 / 0.2642     | 0.4070 / 0.2584     |
| Logistic     | 0.4326 / 0.3704     | 0.4375 / 0.3682     | 0.4328 / 0.3659     | 0.4295 / 0.3528     | 0.4238 / 0.3508     |
| SGD          | 0.4123 / 0.3471     | 0.4106 / 0.3545     | 0.4066 / 0.3338     | 0.4060 / 0.3144     | 0.3952 / 0.3352     |
| RandomForest | **0.4486 / 0.4102** | 0.4484 / **0.4097** | 0.4416 / 0.3997     | 0.4427 / **0.3973** | 0.4330 / 0.3899     |
| ExtraTrees   | 0.4440 / 0.3980     | 0.4449 / 0.3968     | 0.4448 / 0.3948     | 0.4387 / 0.3822     | 0.4333 / 0.3780     |
| XGBoost      | 0.4443 / 0.4102     | 0.4433 / 0.4085     | 0.4429 / **0.4051** | 0.4349 / 0.3946     | 0.4347 / **0.3984** |
| LightGBM     | 0.4468 / 0.3946     | **0.4539 / 0.3975** | **0.4518 / 0.3968** | **0.4447 / 0.3882** | **0.4422 / 0.3859** |
| CatBoost     | 0.4376 / 0.4054     | 0.4388 / 0.3989     | 0.4364 / 0.3947     | 0.4337 / 0.3931     | 0.4284 / 0.3835     |
| NeuralCF     | 0.4146 / 0.2707     | 0.4193 / 0.2760     | 0.4284 / 0.3141     | 0.4232 / 0.2873     | 0.4198 / 0.3590     |
| Simase       | 0.4320 / 0.3225     | 0.4429 / 0.3622     | 0.4422 / 0.3624     | 0.3987 / 0.2557     | 0.4070 / 0.2584     |
| GNN council  | **0.4572 / 0.4200** | **0.4567 / 0.4112** | **0.4522** / 0.3973 | **0.4484 / 0.4020** | **0.4460** / 0.3967 |

> **Bold cells** denote the best score (per metric) in that fold. For each fold I bolded the highest accuracy and the highest macro-F1 value independently.
> I haved made bold to best GCN and ML models both to see difference 
---

## Aggregated Averages (across 5 folds)

| Model        | Mean Accuracy | Mean Macro-F1 |
| ------------ | ------------: | ------------: |
| Baseline     |       0.40746 |       0.26676 |
| Logistic     |       0.43124 |       0.36162 |
| SGD          |       0.40614 |       0.33698 |
| RandomForest |       0.44286 |       0.40136 |
| ExtraTrees   |       0.44114 |       0.38996 |
| XGBoost      |       0.44002 |   **0.40336** |
| LightGBM     |   **0.44788** |       0.39260 |
| CatBoost     |       0.43498 |       0.39512 |
| NeuralCF     |       0.42106 |       0.30142 |
| Simase       |       0.41920 |       0.33280 |
| GNN council  |   **0.45213** |   **0.40545** |

* **Best average accuracy:** **GNN Council (0.45213)**
* **Best average macro-F1:** **GNN Council (0.40545)**

---

---

## Key Observations

1. Matrix factorization + tree ensembles perform best.
2. Pure bias baseline ≈ 40% accuracy.
3. NeuralCF improves over baseline but does not outperform boosted trees.
4. Siamese architecture provides stable but slightly lower performance than boosting.
5. Council model performs best 

---

## Conclusion

This project demonstrates that:

* Strong collaborative filtering features (SVD + biases) are sufficient.
* Ensemble tree methods achieve ~45% accuracy.
* Deep CF models are competitive but require larger datasets to dominate.



---

## How to Run ML models

```bash
python cf_ass2.py
```
## How to run GNN Council
```bash
python gnntest.py
```


Results saved to:

* results.csv
* saved_models/
* reports/

---

## Author

Aditya Upadhyay

