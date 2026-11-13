# Data

The dataset is not stored in this repository. Download it with:

```bash
python -m credit_risk.data
```

**Source:** Yeh, I-C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository,
dataset id 350. https://doi.org/10.24432/C55S3H — licensed under CC BY 4.0.

30,000 credit card clients in Taiwan (April–September 2005): credit limit, demographics,
six months of repayment status, bill amounts and payments, and whether the client defaulted
the following month.

For offline testing, `python -m credit_risk.data --synthetic` writes a small synthetic dataset
with the same columns.
