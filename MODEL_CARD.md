# Model Card: Credit Default Scoring Model

## Overview

| | |
|---|---|
| **Model** | Gradient-boosted trees (XGBoost), logistic regression baseline |
| **Version** | 1.0.0 |
| **Output** | Probability that a credit card client defaults next month, risk band A–E, decision, top risk drivers |
| **Owner** | Pooja Adhav |
| **Status** | Portfolio / research project, not used for real lending decisions |

## Intended use

Support, not replace, a credit analyst when reviewing credit card accounts: prioritise
accounts for review, explain which factors drive the risk, and route borderline cases to
manual review.

**Out of scope:** fully automated decisions on real customers, other credit products, or
populations different from the training data (Taiwan, 2005).

## Data

UCI *Default of Credit Card Clients* (30,000 clients, CC BY 4.0). Split 60 / 20 / 20 into
train, validation (early stopping and threshold choice) and test (final reporting only).
See [data/README.md](data/README.md).

## Features

Credit limit, education, age, six months of repayment status, bills and payments, plus
engineered ratios (utilisation, repayment ratio, number of late months, worst delay, balance
trend). **Sex and marital status are excluded from the model** and used only to monitor fairness.

## Evaluation

Metrics on the held-out test set are written to `reports/metrics.json` on every training run:
AUC, Gini, KS, Brier score and log loss, plus precision, recall and expected cost at the chosen
threshold. The threshold minimises expected cost assuming a missed defaulter costs 5× a
wrongly declined client; change `COST_FALSE_NEGATIVE` in `config.py` to reflect a different
risk appetite.

## Fairness

For sex and age band the training run reports approval rate, true-positive rate and
false-positive rate per group, and the ratio between the lowest and highest approval rate
(the "four-fifths rule" flags ratios below 0.8). Differences in approval rate partly reflect
real differences in default rates, so they need human interpretation rather than automatic
correction.

## Explainability

Each prediction returns its top risk drivers, computed as SHAP values directly by XGBoost
(`pred_contribs`), described in plain language.

## Regulatory context (EU)

- **EU AI Act:** AI systems used to evaluate the creditworthiness of natural persons are
  classified as **high-risk** (Annex III). A production version would need a risk management
  system, data governance, technical documentation, logging, human oversight and accuracy
  and robustness monitoring. This card, the fairness report and the reason codes are a
  starting point for that documentation.
- **GDPR Art. 22:** individuals should not be subject to purely automated decisions with
  significant effects. The `manual_review` band and the reason codes support human oversight.
- **EBA Guidelines on loan origination and monitoring** expect institutions to understand
  and explain the models they use.

## Limitations

- Data is from 2005 and one country. Behaviour and macroeconomic conditions differ today.
- No income, employment or bureau data, which real scorecards rely on.
- Probabilities should be re-calibrated before use on a new population.
- No monitoring for data drift is implemented yet (see roadmap in the README).
