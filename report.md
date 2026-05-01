# Ship Fuel Prediction — Model Analysis Report

## Part 1: Why Does Linear Regression NOT Overfit on Non-Linear Data?

### The Short Answer

Linear Regression doesn't overfit here because **the data is overwhelmingly linear**, and you have **92,567 rows** — far more data points than model parameters. Overfitting occurs when a model has too many parameters relative to training samples. With 74,053 training rows and only 10 parameters (9 features + intercept), overfitting is mathematically almost impossible.

---

### Evidence 1: The Data Is Actually ~96% Linear

The Pearson correlation coefficients reveal that the two dominant features — **Shaft Power** and **RPM** — have extremely strong *linear* correlations with fuel consumption:

| Feature | Pearson r | Interpretation |
|---|---|---|
| **Main engine shaft power (KW)** | **0.9785** | Near-perfect linear relationship |
| **Main engine shaft RPM** | **0.8699** | Very strong linear |
| Speed through water (kn) | 0.6601 | Moderate linear |
| Mean draft (m) | 0.4105 | Weak-moderate |
| Trim (m) | -0.3917 | Weak negative |
| Wave height (m) | 0.2340 | Weak |
| Swell height (m) | 0.2267 | Weak |
| Wind speed (m/s) | 0.2161 | Weak |
| Current speed (m/s) | 0.0375 | Negligible |

> [!IMPORTANT]
> A plain Linear Regression model using **only RPM and Power** achieves **R² = 0.9573** — explaining 95.7% of variance with just 2 features and a straight line. The remaining 7 features collectively add only ~1% more explanatory power.

This means the "non-linearity" in the data contributes less than **4% of the total variance**. Linear Regression captures the dominant linear component perfectly and simply ignores the small non-linear residuals.

---

### Evidence 2: Polynomial Degree Sweep

To test whether adding non-linear capacity helps (or causes overfitting), we trained Linear Regression with PolynomialFeatures at degrees 1–4:

| Degree | Parameters | Train R² | Test R² | Gap (Train−Test) | Test MAE |
|---|---|---|---|---|---|
| 1 (plain linear) | 10 | 0.963056 | 0.962907 | 0.000149 | 27.18 L/hr |
| 2 (quadratic) | 55 | 0.968898 | 0.969084 | -0.000186 | 24.77 L/hr |
| 3 (cubic) | 220 | 0.975304 | 0.974980 | 0.000324 | 23.18 L/hr |
| 4 (quartic) | 715 | 0.980124 | 0.978591 | 0.001533 | 21.52 L/hr |

> [!NOTE]
> Even at **degree 4 with 715 parameters**, the train-test gap is only **0.0015** — no overfitting. This is because 74,053 training samples ÷ 715 parameters = **103 samples per parameter**, which is more than sufficient.

The rule of thumb is: **overfitting starts when samples-per-parameter drops below ~10–20**. With 74k rows, you'd need approximately **3,700+ parameters** before overfitting appears.

---

### Evidence 3: Sample Size Controls Overfitting

The clearest proof — **overfitting IS possible**, but only when you starve the model of data. Here's what happens when we train the degree-3 polynomial (220 parameters) on progressively fewer rows:

| Training Samples | Samples/Param | Train R² | Test R² | Gap | Overfit? |
|---|---|---|---|---|---|
| 100 | 0.5 | **1.000000** | **-0.826690** | 1.8267 | ⚠️ **Severe** |
| 500 | 2.3 | 0.993896 | 0.343562 | 0.6503 | ⚠️ **Severe** |
| 1,000 | 4.5 | 0.989310 | 0.921797 | 0.0675 | ⚠️ **Yes** |
| 5,000 | 22.7 | 0.980490 | 0.971400 | 0.0091 | ✅ No |
| 10,000 | 45.5 | 0.977318 | 0.973958 | 0.0034 | ✅ No |
| 30,000 | 136.4 | 0.975166 | 0.974670 | 0.0005 | ✅ No |
| **74,053** | **336.6** | **0.975304** | **0.974980** | **0.0003** | ✅ **No** |

> [!CAUTION]
> At **n=100**, the model achieves a PERFECT Train R² of 1.0 but a **negative** Test R² of -0.83 — classic catastrophic overfitting. The model memorized 100 training points perfectly but learned nothing generalizable.

This table is the key insight: **overfitting is not a property of the model alone — it's a function of the model complexity relative to the training data size.** With 74,053 rows and 220 parameters, the samples-per-parameter ratio is 337:1. The model cannot possibly memorize the data.

---

### Evidence 4: Residual Analysis

The residuals from plain Linear Regression (degree 1) tell us how much non-linearity the model is missing:

| Metric | Value |
|---|---|
| Residual mean | -0.82 L/hr |
| Residual std | 43.39 L/hr |
| Residual range | [-772.37, 597.14] L/hr |
| Residual as % of target range | **3.1%** |

The residual standard deviation is only **3.1% of the target range**. This means the non-linear patterns that Linear Regression *cannot* capture account for a very small portion of the total variation. The model is not "failing on non-linear data" — it's capturing the dominant linear signal effectively and leaving behind only noise-level residuals.

---

### Summary: Why No Overfitting

```
Overfitting = f(model_complexity, data_size, noise_level)
```

In this case:
1. **Model complexity is LOW** — Even degree-3 polynomial has only 220 params
2. **Data size is HUGE** — 74,053 training rows (337 samples per parameter)
3. **Signal is LINEAR** — Power alone has r=0.98 with fuel consumption
4. **Non-linear component is SMALL** — Only 3.1% of variance is non-linear

All four factors conspire to make overfitting mathematically impossible in this setting.

---

## Part 2: Effect of Synthetic Data Augmentation on Model Accuracy

### Experimental Setup

To measure the effect of augmentation, we compare two training regimes tested on the **same held-out real-data test set** (16,114 rows):

| Split | Rows | Composition |
|---|---|---|
| Real-only Train | 64,453 | 100% real telemetry |
| Augmented Train | 76,453 | 64,453 real + 12,000 synthetic |
| Test Set | 16,114 | 100% real telemetry (always) |

### Results: Augmentation Reduces Accuracy

#### R² Score Comparison (Test Set — Real Data Only)

| Model | Real-Only R² | Augmented R² | Change | Direction |
|---|---|---|---|---|
| Linear Regression | 0.970641 | 0.966919 | **-0.003722** | 📉 Worse |
| Random Forest | 0.982699 | 0.981855 | **-0.000844** | 📉 Worse |
| XGBoost | 0.983121 | 0.982190 | **-0.000931** | 📉 Worse |

#### MAE Comparison (Test Set — Real Data Only)

| Model | Real-Only MAE | Augmented MAE | Change | Direction |
|---|---|---|---|---|
| Linear Regression | 21.08 L/hr | 23.40 L/hr | **+2.32 L/hr** | 📉 Worse |
| Random Forest | 17.48 L/hr | 17.82 L/hr | **+0.34 L/hr** | 📉 Worse |
| XGBoost | 17.51 L/hr | 17.70 L/hr | **+0.19 L/hr** | 📉 Worse |

#### Full Metrics Breakdown

| Model | Training | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| Linear Regression | Real-only | 0.9706 | 21.08 | 35.53 | 2.61% |
| Linear Regression | Augmented | 0.9669 | 23.40 | 37.72 | 2.87% |
| Random Forest | Real-only | 0.9827 | 17.48 | 27.28 | 2.08% |
| Random Forest | Augmented | 0.9819 | 17.82 | 27.94 | 2.12% |
| XGBoost | Real-only | 0.9831 | 17.51 | 26.94 | 2.13% |
| XGBoost | Augmented | 0.9822 | 17.70 | 27.68 | 2.12% |

> [!WARNING]
> **All three models perform worse when trained with synthetic augmentation.** The degradation is most severe for Linear Regression (-0.37% R², +2.32 L/hr MAE) and least severe for XGBoost (-0.09% R², +0.19 L/hr MAE).

### Why Does Augmentation Hurt?

1. **Distribution mismatch**: Synthetic data follows a slightly different distribution than real telemetry. When the model trains on both, it learns a "blended" decision boundary that doesn't perfectly fit either distribution. When evaluated on real-only test data, this mismatch shows up as increased error.

2. **Signal dilution**: With 64,453 real rows already providing excellent coverage, adding 12,000 synthetic rows doesn't fill any gaps in the feature space — it instead dilutes the real signal. The models are already well-trained on the real data.

3. **Noise injection**: Synthetic data generation (e.g., via SMOTE, Gaussian noise, or GAN) introduces systematic biases. These biases are learned by the model and degrade test-set predictions.

4. **Linear Regression suffers most**: Because LR computes a global best-fit plane, synthetic outliers shift the entire plane. Tree-based models (RF, XGB) are more robust because they make local decisions — a bad synthetic point only affects the leaf it falls into.

> [!TIP]
> Augmentation is beneficial when you have very few real samples (e.g., <1,000 rows) and the synthetic data helps the model generalize. In this case, with 80,567 real records, the dataset is already large enough that augmentation provides no benefit — it only introduces noise.
