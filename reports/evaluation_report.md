# Phase 6 — Evaluation Report

**Protocol:** Leave-One-Bearing-Out (LOBO) CV, inherited verbatim from Phase 5 fold assignments.
**Near-failure threshold:** RUL ≤ 1440 minutes.

> **Honesty guardrail (architecture §10.4):** every metric below is the average/spread of **exactly 4 evaluation runs (n=4 units)**. These numbers carry **no classical confidence interval** and are **methodology validation, not fleet accuracy**.

---

## 4-Fold Summary (all models)

| Model | Mean MAE (min) | Fold Spread (min) | Stability Ratio | Mean Signed Error | Late-Prediction Rate |
|-------|---------------:|-------------------:|-----------------:|-------------------:|----------------------:|
| mean | 16538.85 | 6200.40 | 1.46 | +4262.31 | 67.6% |
| median | 16149.83 | 8798.01 | 1.69 | +1445.97 | 60.6% |
| linear | 18125.03 | 12347.15 | 2.02 | +1152.11 | 58.0% |
| random_forest | 13218.45 | 7564.25 | 1.77 | -3524.89 | 41.5% |
| xgboost | 12589.37 | 8031.97 | 2.05 | -2295.71 | 45.6% |

---

## Per-Fold Detail

### mean

| Fold | Train Bearing | Test Bearing | MAE | RMSE | R² | Near-Failure MAE | Train MAE | Overfitting Ratio |
|------|---------------|--------------|----:|-----:|---:|------------------:|----------:|-------------------:|
| fold_1 | set1::bearing_4, set2::bearing_1, set3::bearing_3 | set1::bearing_3 | 13528.71 | 15320.79 | -0.180 | 24451.13 | 15892.45 | 1.17 |
| fold_2 | set1::bearing_3, set2::bearing_1, set3::bearing_3 | set1::bearing_4 | 13528.71 | 15320.79 | -0.180 | 24451.13 | 15892.45 | 1.17 |
| fold_3 | set1::bearing_3, set1::bearing_4, set3::bearing_3 | set2::bearing_1 | 19729.11 | 19932.55 | -48.240 | 23924.11 | 14874.94 | 0.75 |
| fold_4 | set1::bearing_3, set1::bearing_4, set2::bearing_1 | set3::bearing_3 | 19368.86 | 23643.99 | -0.622 | 16895.36 | 11951.91 | 0.62 |

### median

| Fold | Train Bearing | Test Bearing | MAE | RMSE | R² | Near-Failure MAE | Train MAE | Overfitting Ratio |
|------|---------------|--------------|----:|-----:|---:|------------------:|----------:|-------------------:|
| fold_1 | set1::bearing_4, set2::bearing_1, set3::bearing_3 | set1::bearing_3 | 12717.44 | 14520.69 | -0.060 | 21918.13 | 15739.33 | 1.24 |
| fold_2 | set1::bearing_3, set2::bearing_1, set3::bearing_3 | set1::bearing_4 | 12717.44 | 14520.69 | -0.060 | 21918.13 | 15739.33 | 1.24 |
| fold_3 | set1::bearing_3, set1::bearing_4, set3::bearing_3 | set2::bearing_1 | 17648.97 | 17876.10 | -38.604 | 21843.97 | 14745.25 | 0.84 |
| fold_4 | set1::bearing_3, set1::bearing_4, set2::bearing_1 | set3::bearing_3 | 21515.46 | 26393.31 | -1.021 | 12776.17 | 11291.30 | 0.52 |

### linear

| Fold | Train Bearing | Test Bearing | MAE | RMSE | R² | Near-Failure MAE | Train MAE | Overfitting Ratio |
|------|---------------|--------------|----:|-----:|---:|------------------:|----------:|-------------------:|
| fold_1 | set1::bearing_4, set2::bearing_1, set3::bearing_3 | set1::bearing_3 | 12113.96 | 40692.03 | -7.322 | 111226.51 | 11938.30 | 0.99 |
| fold_2 | set1::bearing_3, set2::bearing_1, set3::bearing_3 | set1::bearing_4 | 15397.30 | 18649.21 | -0.748 | 4886.59 | 12479.22 | 0.81 |
| fold_3 | set1::bearing_3, set1::bearing_4, set3::bearing_3 | set2::bearing_1 | 20527.74 | 22072.03 | -59.378 | 13159.36 | 11252.45 | 0.55 |
| fold_4 | set1::bearing_3, set1::bearing_4, set2::bearing_1 | set3::bearing_3 | 24461.11 | 29804.36 | -1.577 | 8482.87 | 8409.32 | 0.34 |

### random_forest

| Fold | Train Bearing | Test Bearing | MAE | RMSE | R² | Near-Failure MAE | Train MAE | Overfitting Ratio |
|------|---------------|--------------|----:|-----:|---:|------------------:|----------:|-------------------:|
| fold_1 | set1::bearing_4, set2::bearing_1, set3::bearing_3 | set1::bearing_3 | 13717.23 | 17806.41 | -0.594 | 1246.94 | 3560.21 | 0.26 |
| fold_2 | set1::bearing_3, set2::bearing_1, set3::bearing_3 | set1::bearing_4 | 9806.78 | 13829.25 | 0.039 | 501.95 | 4684.34 | 0.48 |
| fold_3 | set1::bearing_3, set1::bearing_4, set3::bearing_3 | set2::bearing_1 | 11978.75 | 16084.09 | -31.062 | 1581.01 | 4398.41 | 0.37 |
| fold_4 | set1::bearing_3, set1::bearing_4, set2::bearing_1 | set3::bearing_3 | 17371.03 | 21451.54 | -0.335 | 213.35 | 3467.02 | 0.20 |

### xgboost

| Fold | Train Bearing | Test Bearing | MAE | RMSE | R² | Near-Failure MAE | Train MAE | Overfitting Ratio |
|------|---------------|--------------|----:|-----:|---:|------------------:|----------:|-------------------:|
| fold_1 | set1::bearing_4, set2::bearing_1, set3::bearing_3 | set1::bearing_3 | 14958.93 | 19306.03 | -0.873 | 2487.28 | 3698.95 | 0.25 |
| fold_2 | set1::bearing_3, set2::bearing_1, set3::bearing_3 | set1::bearing_4 | 7681.20 | 10155.62 | 0.482 | 1913.01 | 4654.47 | 0.61 |
| fold_3 | set1::bearing_3, set1::bearing_4, set3::bearing_3 | set2::bearing_1 | 12004.20 | 14967.28 | -26.764 | 2353.10 | 4384.26 | 0.37 |
| fold_4 | set1::bearing_3, set1::bearing_4, set2::bearing_1 | set3::bearing_3 | 15713.16 | 18627.32 | -0.006 | 1545.94 | 2940.77 | 0.19 |

---

## Error Analysis (architecture §10.6)

### mean

**fold_1** (test bearing: set1::bearing_3)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +5979.49 min
- Late-prediction rate: 71.6%
- Channel-specific MAE: channel 5: MAE=13528.71, channel 6: MAE=13528.71
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_3 | 5 | 2003-11-25 23:39:56 | 0.00 | 25043.97 | 25043.97 |
  | set1::bearing_3 | 6 | 2003-11-25 23:39:56 | 0.00 | 25043.97 | 25043.97 |
  | set1::bearing_3 | 6 | 2003-11-25 23:29:56 | 10.00 | 25043.97 | 25033.97 |
  | set1::bearing_3 | 5 | 2003-11-25 23:29:56 | 10.00 | 25043.97 | 25033.97 |
  | set1::bearing_3 | 5 | 2003-11-25 23:19:56 | 20.00 | 25043.97 | 25023.97 |
  | set1::bearing_3 | 6 | 2003-11-25 23:19:56 | 20.00 | 25043.97 | 25023.97 |
  | set1::bearing_3 | 6 | 2003-11-25 23:13:21 | 26.58 | 25043.97 | 25017.38 |
  | set1::bearing_3 | 5 | 2003-11-25 23:13:21 | 26.58 | 25043.97 | 25017.38 |
  | set1::bearing_3 | 6 | 2003-10-22 12:06:24 | 49653.53 | 25043.97 | 24609.57 |
  | set1::bearing_3 | 5 | 2003-10-22 12:06:24 | 49653.53 | 25043.97 | 24609.57 |

**fold_2** (test bearing: set1::bearing_4)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +5979.49 min
- Late-prediction rate: 71.6%
- Channel-specific MAE: channel 7: MAE=13528.71, channel 8: MAE=13528.71
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_4 | 7 | 2003-11-25 23:39:56 | 0.00 | 25043.97 | 25043.97 |
  | set1::bearing_4 | 8 | 2003-11-25 23:39:56 | 0.00 | 25043.97 | 25043.97 |
  | set1::bearing_4 | 8 | 2003-11-25 23:29:56 | 10.00 | 25043.97 | 25033.97 |
  | set1::bearing_4 | 7 | 2003-11-25 23:29:56 | 10.00 | 25043.97 | 25033.97 |
  | set1::bearing_4 | 7 | 2003-11-25 23:19:56 | 20.00 | 25043.97 | 25023.97 |
  | set1::bearing_4 | 8 | 2003-11-25 23:19:56 | 20.00 | 25043.97 | 25023.97 |
  | set1::bearing_4 | 8 | 2003-11-25 23:13:21 | 26.58 | 25043.97 | 25017.38 |
  | set1::bearing_4 | 7 | 2003-11-25 23:13:21 | 26.58 | 25043.97 | 25017.38 |
  | set1::bearing_4 | 8 | 2003-10-22 12:06:24 | 49653.53 | 25043.97 | 24609.57 |
  | set1::bearing_4 | 7 | 2003-10-22 12:06:24 | 49653.53 | 25043.97 | 24609.57 |

**fold_3** (test bearing: set2::bearing_1)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +19729.11 min
- Late-prediction rate: 100.0%
- Channel-specific MAE: channel 1: MAE=19729.11
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set2::bearing_1 | 1 | 2004-02-19 06:22:39 | 0.00 | 24644.11 | 24644.11 |
  | set2::bearing_1 | 1 | 2004-02-19 06:12:39 | 10.00 | 24644.11 | 24634.11 |
  | set2::bearing_1 | 1 | 2004-02-19 06:02:39 | 20.00 | 24644.11 | 24624.11 |
  | set2::bearing_1 | 1 | 2004-02-19 05:52:39 | 30.00 | 24644.11 | 24614.11 |
  | set2::bearing_1 | 1 | 2004-02-19 05:42:39 | 40.00 | 24644.11 | 24604.11 |
  | set2::bearing_1 | 1 | 2004-02-19 05:32:39 | 50.00 | 24644.11 | 24594.11 |
  | set2::bearing_1 | 1 | 2004-02-19 05:22:39 | 60.00 | 24644.11 | 24584.11 |
  | set2::bearing_1 | 1 | 2004-02-19 05:12:39 | 70.00 | 24644.11 | 24574.11 |
  | set2::bearing_1 | 1 | 2004-02-19 05:02:39 | 80.00 | 24644.11 | 24564.11 |
  | set2::bearing_1 | 1 | 2004-02-19 04:52:39 | 90.00 | 24644.11 | 24554.11 |

**fold_4** (test bearing: set3::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -14638.87 min
- Late-prediction rate: 27.2%
- Channel-specific MAE: channel 3: MAE=19368.86
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set3::bearing_3 | 3 | 2004-03-04 09:27:46 | 64385.15 | 17615.36 | 46769.79 |
  | set3::bearing_3 | 3 | 2004-03-04 09:32:46 | 64380.15 | 17615.36 | 46764.79 |
  | set3::bearing_3 | 3 | 2004-03-04 09:42:46 | 64370.15 | 17615.36 | 46754.79 |
  | set3::bearing_3 | 3 | 2004-03-04 09:52:46 | 64360.15 | 17615.36 | 46744.79 |
  | set3::bearing_3 | 3 | 2004-03-04 10:02:46 | 64350.15 | 17615.36 | 46734.79 |
  | set3::bearing_3 | 3 | 2004-03-04 10:12:46 | 64340.15 | 17615.36 | 46724.79 |
  | set3::bearing_3 | 3 | 2004-03-04 10:22:46 | 64330.15 | 17615.36 | 46714.79 |
  | set3::bearing_3 | 3 | 2004-03-04 10:32:46 | 64320.15 | 17615.36 | 46704.79 |
  | set3::bearing_3 | 3 | 2004-03-04 10:42:46 | 64310.15 | 17615.36 | 46694.79 |
  | set3::bearing_3 | 3 | 2004-03-04 10:52:46 | 64300.15 | 17615.36 | 46684.79 |

### median

**fold_1** (test bearing: set1::bearing_3)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +3446.49 min
- Late-prediction rate: 60.8%
- Channel-specific MAE: channel 5: MAE=12717.44, channel 6: MAE=12717.44
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_3 | 5 | 2003-10-22 12:06:24 | 49653.53 | 22510.97 | 27142.57 |
  | set1::bearing_3 | 6 | 2003-10-22 12:06:24 | 49653.53 | 22510.97 | 27142.57 |
  | set1::bearing_3 | 6 | 2003-10-22 12:09:13 | 49650.72 | 22510.97 | 27139.75 |
  | set1::bearing_3 | 5 | 2003-10-22 12:09:13 | 49650.72 | 22510.97 | 27139.75 |
  | set1::bearing_3 | 5 | 2003-10-22 12:14:13 | 49645.72 | 22510.97 | 27134.75 |
  | set1::bearing_3 | 6 | 2003-10-22 12:14:13 | 49645.72 | 22510.97 | 27134.75 |
  | set1::bearing_3 | 6 | 2003-10-22 12:19:13 | 49640.72 | 22510.97 | 27129.75 |
  | set1::bearing_3 | 5 | 2003-10-22 12:19:13 | 49640.72 | 22510.97 | 27129.75 |
  | set1::bearing_3 | 6 | 2003-10-22 12:24:13 | 49635.72 | 22510.97 | 27124.75 |
  | set1::bearing_3 | 5 | 2003-10-22 12:24:13 | 49635.72 | 22510.97 | 27124.75 |

**fold_2** (test bearing: set1::bearing_4)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +3446.49 min
- Late-prediction rate: 60.8%
- Channel-specific MAE: channel 7: MAE=12717.44, channel 8: MAE=12717.44
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_4 | 7 | 2003-10-22 12:06:24 | 49653.53 | 22510.97 | 27142.57 |
  | set1::bearing_4 | 8 | 2003-10-22 12:06:24 | 49653.53 | 22510.97 | 27142.57 |
  | set1::bearing_4 | 8 | 2003-10-22 12:09:13 | 49650.72 | 22510.97 | 27139.75 |
  | set1::bearing_4 | 7 | 2003-10-22 12:09:13 | 49650.72 | 22510.97 | 27139.75 |
  | set1::bearing_4 | 7 | 2003-10-22 12:14:13 | 49645.72 | 22510.97 | 27134.75 |
  | set1::bearing_4 | 8 | 2003-10-22 12:14:13 | 49645.72 | 22510.97 | 27134.75 |
  | set1::bearing_4 | 8 | 2003-10-22 12:19:13 | 49640.72 | 22510.97 | 27129.75 |
  | set1::bearing_4 | 7 | 2003-10-22 12:19:13 | 49640.72 | 22510.97 | 27129.75 |
  | set1::bearing_4 | 8 | 2003-10-22 12:24:13 | 49635.72 | 22510.97 | 27124.75 |
  | set1::bearing_4 | 7 | 2003-10-22 12:24:13 | 49635.72 | 22510.97 | 27124.75 |

**fold_3** (test bearing: set2::bearing_1)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +17648.97 min
- Late-prediction rate: 100.0%
- Channel-specific MAE: channel 1: MAE=17648.97
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set2::bearing_1 | 1 | 2004-02-19 06:22:39 | 0.00 | 22563.97 | 22563.97 |
  | set2::bearing_1 | 1 | 2004-02-19 06:12:39 | 10.00 | 22563.97 | 22553.97 |
  | set2::bearing_1 | 1 | 2004-02-19 06:02:39 | 20.00 | 22563.97 | 22543.97 |
  | set2::bearing_1 | 1 | 2004-02-19 05:52:39 | 30.00 | 22563.97 | 22533.97 |
  | set2::bearing_1 | 1 | 2004-02-19 05:42:39 | 40.00 | 22563.97 | 22523.97 |
  | set2::bearing_1 | 1 | 2004-02-19 05:32:39 | 50.00 | 22563.97 | 22513.97 |
  | set2::bearing_1 | 1 | 2004-02-19 05:22:39 | 60.00 | 22563.97 | 22503.97 |
  | set2::bearing_1 | 1 | 2004-02-19 05:12:39 | 70.00 | 22563.97 | 22493.97 |
  | set2::bearing_1 | 1 | 2004-02-19 05:02:39 | 80.00 | 22563.97 | 22483.97 |
  | set2::bearing_1 | 1 | 2004-02-19 04:52:39 | 90.00 | 22563.97 | 22473.97 |

**fold_4** (test bearing: set3::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -18758.06 min
- Late-prediction rate: 20.7%
- Channel-specific MAE: channel 3: MAE=21515.46
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set3::bearing_3 | 3 | 2004-03-04 09:27:46 | 64385.15 | 13496.17 | 50888.98 |
  | set3::bearing_3 | 3 | 2004-03-04 09:32:46 | 64380.15 | 13496.17 | 50883.98 |
  | set3::bearing_3 | 3 | 2004-03-04 09:42:46 | 64370.15 | 13496.17 | 50873.98 |
  | set3::bearing_3 | 3 | 2004-03-04 09:52:46 | 64360.15 | 13496.17 | 50863.98 |
  | set3::bearing_3 | 3 | 2004-03-04 10:02:46 | 64350.15 | 13496.17 | 50853.98 |
  | set3::bearing_3 | 3 | 2004-03-04 10:12:46 | 64340.15 | 13496.17 | 50843.98 |
  | set3::bearing_3 | 3 | 2004-03-04 10:22:46 | 64330.15 | 13496.17 | 50833.98 |
  | set3::bearing_3 | 3 | 2004-03-04 10:32:46 | 64320.15 | 13496.17 | 50823.98 |
  | set3::bearing_3 | 3 | 2004-03-04 10:42:46 | 64310.15 | 13496.17 | 50813.98 |
  | set3::bearing_3 | 3 | 2004-03-04 10:52:46 | 64300.15 | 13496.17 | 50803.98 |

### linear

**fold_1** (test bearing: set1::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -4054.14 min
- Late-prediction rate: 38.6%
- Channel-specific MAE: channel 5: MAE=12754.48, channel 6: MAE=11473.44
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_3 | 5 | 2003-11-25 23:39:56 | 0.00 | 1695661.03 | 1695661.03 |
  | set1::bearing_3 | 5 | 2003-11-25 23:19:56 | 20.00 | 1070028.26 | 1070008.26 |
  | set1::bearing_3 | 5 | 2003-11-25 23:29:56 | 10.00 | 785725.79 | 785715.79 |
  | set1::bearing_3 | 6 | 2003-11-25 23:39:56 | 0.00 | 783197.96 | 783197.96 |
  | set1::bearing_3 | 5 | 2003-11-25 23:13:21 | 26.58 | 729948.88 | 729922.30 |
  | set1::bearing_3 | 6 | 2003-11-25 23:19:56 | 20.00 | 464924.38 | 464904.38 |
  | set1::bearing_3 | 6 | 2003-11-25 23:29:56 | 10.00 | 338536.46 | 338526.46 |
  | set1::bearing_3 | 6 | 2003-11-25 23:13:21 | 26.58 | 322093.66 | 322067.07 |
  | set1::bearing_3 | 5 | 2003-11-25 15:07:32 | 512.40 | 114158.24 | 113645.84 |
  | set1::bearing_3 | 5 | 2003-11-25 13:37:32 | 602.40 | 95902.87 | 95300.47 |

**fold_2** (test bearing: set1::bearing_4)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +13560.01 min
- Late-prediction rate: 86.5%
- Channel-specific MAE: channel 7: MAE=17241.78, channel 8: MAE=13552.82
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_4 | 7 | 2003-10-22 14:04:13 | 49535.72 | -17251.29 | 66787.00 |
  | set1::bearing_4 | 8 | 2003-10-29 15:39:46 | 39360.17 | -5570.61 | 44930.77 |
  | set1::bearing_4 | 7 | 2003-11-19 15:46:07 | 9113.82 | 49688.68 | 40574.86 |
  | set1::bearing_4 | 7 | 2003-11-16 14:08:46 | 13531.17 | 48903.99 | 35372.82 |
  | set1::bearing_4 | 7 | 2003-11-15 09:08:46 | 15271.17 | 50544.18 | 35273.01 |
  | set1::bearing_4 | 7 | 2003-11-16 14:18:46 | 13521.17 | 48472.84 | 34951.67 |
  | set1::bearing_4 | 7 | 2003-11-16 20:28:46 | 13151.17 | 47976.64 | 34825.48 |
  | set1::bearing_4 | 7 | 2003-11-16 10:08:46 | 13771.17 | 48503.97 | 34732.80 |
  | set1::bearing_4 | 7 | 2003-11-16 13:08:46 | 13591.17 | 48196.11 | 34604.94 |
  | set1::bearing_4 | 7 | 2003-11-16 10:18:46 | 13761.17 | 48105.42 | 34344.26 |

**fold_3** (test bearing: set2::bearing_1)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +18628.69 min
- Late-prediction rate: 95.1%
- Channel-specific MAE: channel 1: MAE=20527.74
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set2::bearing_1 | 1 | 2004-02-19 06:22:39 | 0.00 | -168394.21 | 168394.21 |
  | set2::bearing_1 | 1 | 2004-02-19 05:42:39 | 40.00 | -84217.31 | 84257.31 |
  | set2::bearing_1 | 1 | 2004-02-19 05:32:39 | 50.00 | -68173.54 | 68223.54 |
  | set2::bearing_1 | 1 | 2004-02-19 06:02:39 | 20.00 | -65692.29 | 65712.29 |
  | set2::bearing_1 | 1 | 2004-02-19 06:12:39 | 10.00 | -61089.95 | 61099.95 |
  | set2::bearing_1 | 1 | 2004-02-19 05:52:39 | 30.00 | -56961.22 | 56991.22 |
  | set2::bearing_1 | 1 | 2004-02-19 05:22:39 | 60.00 | -56054.08 | 56114.08 |
  | set2::bearing_1 | 1 | 2004-02-19 05:12:39 | 70.00 | -39159.02 | 39229.02 |
  | set2::bearing_1 | 1 | 2004-02-16 07:12:39 | 4270.00 | 30885.48 | 26615.48 |
  | set2::bearing_1 | 1 | 2004-02-15 20:02:39 | 4940.00 | 31504.25 | 26564.25 |

**fold_4** (test bearing: set3::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -23526.14 min
- Late-prediction rate: 11.7%
- Channel-specific MAE: channel 3: MAE=24461.11
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set3::bearing_3 | 3 | 2004-04-18 02:22:55 | 10.00 | 79886.73 | 79876.73 |
  | set3::bearing_3 | 3 | 2004-04-18 02:02:55 | 30.00 | 64046.87 | 64016.87 |
  | set3::bearing_3 | 3 | 2004-03-04 13:22:46 | 64150.15 | 1028.46 | 63121.69 |
  | set3::bearing_3 | 3 | 2004-03-04 10:22:46 | 64330.15 | 2821.88 | 61508.27 |
  | set3::bearing_3 | 3 | 2004-03-04 20:22:46 | 63730.15 | 2827.69 | 60902.46 |
  | set3::bearing_3 | 3 | 2004-03-04 14:22:46 | 64090.15 | 3788.51 | 60301.64 |
  | set3::bearing_3 | 3 | 2004-03-04 15:32:46 | 64020.15 | 3813.79 | 60206.36 |
  | set3::bearing_3 | 3 | 2004-03-04 16:42:46 | 63950.15 | 4332.08 | 59618.07 |
  | set3::bearing_3 | 3 | 2004-03-04 16:22:46 | 63970.15 | 4604.45 | 59365.70 |
  | set3::bearing_3 | 3 | 2004-03-04 16:12:46 | 63980.15 | 4783.40 | 59196.75 |

### random_forest

**fold_1** (test bearing: set1::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -13103.32 min
- Late-prediction rate: 13.8%
- Channel-specific MAE: channel 5: MAE=10992.81, channel 6: MAE=16441.66
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_3 | 6 | 2003-10-29 15:09:46 | 39390.17 | 1502.27 | 37887.90 |
  | set1::bearing_3 | 6 | 2003-10-29 15:39:46 | 39360.17 | 1502.27 | 37857.90 |
  | set1::bearing_3 | 6 | 2003-10-29 15:49:46 | 39350.17 | 1502.27 | 37847.90 |
  | set1::bearing_3 | 6 | 2003-10-29 16:19:46 | 39320.17 | 1502.27 | 37817.90 |
  | set1::bearing_3 | 6 | 2003-10-29 16:29:46 | 39310.17 | 1514.84 | 37795.33 |
  | set1::bearing_3 | 6 | 2003-10-29 16:49:46 | 39290.17 | 1502.27 | 37787.90 |
  | set1::bearing_3 | 6 | 2003-10-29 16:59:46 | 39280.17 | 1502.27 | 37777.90 |
  | set1::bearing_3 | 6 | 2003-10-29 17:29:46 | 39250.17 | 1502.27 | 37747.90 |
  | set1::bearing_3 | 6 | 2003-10-29 17:49:46 | 39230.17 | 1502.27 | 37727.90 |
  | set1::bearing_3 | 6 | 2003-10-29 17:59:46 | 39220.17 | 1502.27 | 37717.90 |

**fold_2** (test bearing: set1::bearing_4)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -543.87 min
- Late-prediction rate: 42.0%
- Channel-specific MAE: channel 7: MAE=9437.20, channel 8: MAE=10176.35
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_4 | 7 | 2003-10-22 13:39:13 | 49560.72 | 5884.56 | 43676.16 |
  | set1::bearing_4 | 7 | 2003-10-22 12:34:13 | 49625.72 | 7948.49 | 41677.23 |
  | set1::bearing_4 | 7 | 2003-10-22 12:44:13 | 49615.72 | 10642.93 | 38972.79 |
  | set1::bearing_4 | 7 | 2003-10-22 14:04:13 | 49535.72 | 10661.06 | 38874.66 |
  | set1::bearing_4 | 7 | 2003-10-22 12:49:13 | 49610.72 | 10759.37 | 38851.35 |
  | set1::bearing_4 | 7 | 2003-10-22 12:59:13 | 49600.72 | 10833.09 | 38767.62 |
  | set1::bearing_4 | 7 | 2003-10-22 15:04:13 | 49475.72 | 10931.07 | 38544.65 |
  | set1::bearing_4 | 7 | 2003-10-22 12:14:13 | 49645.72 | 11600.91 | 38044.81 |
  | set1::bearing_4 | 7 | 2003-10-22 12:06:24 | 49653.53 | 11610.63 | 38042.91 |
  | set1::bearing_4 | 7 | 2003-10-22 15:29:13 | 49450.72 | 11411.22 | 38039.50 |

**fold_3** (test bearing: set2::bearing_1)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +11605.40 min
- Late-prediction rate: 79.3%
- Channel-specific MAE: channel 1: MAE=11978.75
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set2::bearing_1 | 1 | 2004-02-19 06:12:39 | 10.00 | 53417.45 | 53407.45 |
  | set2::bearing_1 | 1 | 2004-02-19 06:22:39 | 0.00 | 53396.87 | 53396.87 |
  | set2::bearing_1 | 1 | 2004-02-15 19:02:39 | 5000.00 | 39604.72 | 34604.72 |
  | set2::bearing_1 | 1 | 2004-02-15 19:42:39 | 4960.00 | 39381.47 | 34421.47 |
  | set2::bearing_1 | 1 | 2004-02-15 20:22:39 | 4920.00 | 39293.39 | 34373.39 |
  | set2::bearing_1 | 1 | 2004-02-15 21:42:39 | 4840.00 | 39008.35 | 34168.35 |
  | set2::bearing_1 | 1 | 2004-02-15 20:12:39 | 4930.00 | 39052.19 | 34122.19 |
  | set2::bearing_1 | 1 | 2004-02-15 13:22:39 | 5340.00 | 39345.36 | 34005.36 |
  | set2::bearing_1 | 1 | 2004-02-14 22:12:39 | 6250.00 | 40069.20 | 33819.20 |
  | set2::bearing_1 | 1 | 2004-02-15 00:12:39 | 6130.00 | 39927.35 | 33797.35 |

**fold_4** (test bearing: set3::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -12057.79 min
- Late-prediction rate: 30.8%
- Channel-specific MAE: channel 3: MAE=17371.03
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set3::bearing_3 | 3 | 2004-03-04 09:42:46 | 64370.15 | 20660.71 | 43709.44 |
  | set3::bearing_3 | 3 | 2004-03-04 09:32:46 | 64380.15 | 20884.55 | 43495.60 |
  | set3::bearing_3 | 3 | 2004-03-04 11:02:46 | 64290.15 | 20976.27 | 43313.88 |
  | set3::bearing_3 | 3 | 2004-03-04 14:12:46 | 64100.15 | 20868.84 | 43231.31 |
  | set3::bearing_3 | 3 | 2004-03-04 17:02:46 | 63930.15 | 20741.08 | 43189.07 |
  | set3::bearing_3 | 3 | 2004-03-04 13:02:46 | 64170.15 | 20989.03 | 43181.12 |
  | set3::bearing_3 | 3 | 2004-03-04 13:32:46 | 64140.15 | 21006.11 | 43134.04 |
  | set3::bearing_3 | 3 | 2004-03-04 12:12:46 | 64220.15 | 21095.48 | 43124.67 |
  | set3::bearing_3 | 3 | 2004-03-04 16:52:46 | 63940.15 | 20818.92 | 43121.23 |
  | set3::bearing_3 | 3 | 2004-03-04 09:52:46 | 64360.15 | 21306.30 | 43053.85 |

### xgboost

**fold_1** (test bearing: set1::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -14578.17 min
- Late-prediction rate: 11.7%
- Channel-specific MAE: channel 5: MAE=13285.80, channel 6: MAE=16632.07
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_3 | 6 | 2003-10-23 06:24:13 | 48555.72 | 9201.75 | 39353.97 |
  | set1::bearing_3 | 6 | 2003-10-22 23:44:13 | 48955.72 | 9691.70 | 39264.02 |
  | set1::bearing_3 | 6 | 2003-10-22 22:14:13 | 49045.72 | 9971.25 | 39074.47 |
  | set1::bearing_3 | 6 | 2003-10-22 14:09:13 | 49530.72 | 10623.16 | 38907.56 |
  | set1::bearing_3 | 6 | 2003-10-22 14:14:13 | 49525.72 | 10746.74 | 38778.98 |
  | set1::bearing_3 | 6 | 2003-10-22 14:49:13 | 49490.72 | 10774.89 | 38715.83 |
  | set1::bearing_3 | 6 | 2003-10-22 15:29:13 | 49450.72 | 10817.68 | 38633.04 |
  | set1::bearing_3 | 6 | 2003-10-23 02:24:13 | 48795.72 | 10169.83 | 38625.89 |
  | set1::bearing_3 | 6 | 2003-10-22 13:39:13 | 49560.72 | 10939.78 | 38620.94 |
  | set1::bearing_3 | 6 | 2003-10-22 21:44:13 | 49075.72 | 10459.73 | 38615.99 |

**fold_2** (test bearing: set1::bearing_4)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -907.89 min
- Late-prediction rate: 43.6%
- Channel-specific MAE: channel 7: MAE=7873.84, channel 8: MAE=7488.56
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set1::bearing_4 | 7 | 2003-10-22 12:34:13 | 49625.72 | 14257.99 | 35367.72 |
  | set1::bearing_4 | 7 | 2003-10-22 13:39:13 | 49560.72 | 16826.61 | 32734.11 |
  | set1::bearing_4 | 7 | 2003-10-31 16:11:44 | 36448.20 | 8305.67 | 28142.53 |
  | set1::bearing_4 | 7 | 2003-11-01 22:11:44 | 34648.20 | 7815.56 | 26832.64 |
  | set1::bearing_4 | 8 | 2003-11-16 08:18:46 | 13881.17 | 39690.28 | 25809.11 |
  | set1::bearing_4 | 7 | 2003-11-01 09:41:44 | 35398.20 | 9663.77 | 25734.43 |
  | set1::bearing_4 | 7 | 2003-10-31 16:01:44 | 36458.20 | 11100.01 | 25358.19 |
  | set1::bearing_4 | 7 | 2003-10-22 14:24:13 | 49515.72 | 24260.75 | 25254.96 |
  | set1::bearing_4 | 7 | 2003-11-01 09:31:44 | 35408.20 | 10302.15 | 25106.05 |
  | set1::bearing_4 | 7 | 2003-10-22 14:19:13 | 49520.72 | 24635.04 | 24885.68 |

**fold_3** (test bearing: set2::bearing_1)
- Systematic bias: OVER-predicts (late warning, dangerous) — mean signed error = +11611.60 min
- Late-prediction rate: 83.8%
- Channel-specific MAE: channel 1: MAE=12004.20
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set2::bearing_1 | 1 | 2004-02-19 06:12:39 | 10.00 | 69506.43 | 69496.43 |
  | set2::bearing_1 | 1 | 2004-02-19 06:22:39 | 0.00 | 67201.47 | 67201.47 |
  | set2::bearing_1 | 1 | 2004-02-15 20:52:39 | 4890.00 | 29280.10 | 24390.10 |
  | set2::bearing_1 | 1 | 2004-02-16 00:02:39 | 4700.00 | 29025.67 | 24325.67 |
  | set2::bearing_1 | 1 | 2004-02-15 01:02:39 | 6080.00 | 30267.60 | 24187.60 |
  | set2::bearing_1 | 1 | 2004-02-14 19:12:39 | 6430.00 | 30313.48 | 23883.48 |
  | set2::bearing_1 | 1 | 2004-02-16 02:32:39 | 4550.00 | 28431.05 | 23881.05 |
  | set2::bearing_1 | 1 | 2004-02-15 13:52:39 | 5310.00 | 29134.85 | 23824.85 |
  | set2::bearing_1 | 1 | 2004-02-16 01:22:39 | 4620.00 | 28382.30 | 23762.30 |
  | set2::bearing_1 | 1 | 2004-02-15 06:22:39 | 5760.00 | 29459.42 | 23699.42 |

**fold_4** (test bearing: set3::bearing_3)
- Systematic bias: UNDER-predicts (early warning, conservative) — mean signed error = -5308.38 min
- Late-prediction rate: 43.1%
- Channel-specific MAE: channel 3: MAE=15713.16
- Largest residuals (top 10):

  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |
  |---------|---------|-----------|-----------:|---------------:|-------------:|
  | set3::bearing_3 | 3 | 2004-03-04 09:52:46 | 64360.15 | 26520.00 | 37840.15 |
  | set3::bearing_3 | 3 | 2004-03-04 14:42:46 | 64070.15 | 26370.82 | 37699.33 |
  | set3::bearing_3 | 3 | 2004-03-04 12:02:46 | 64230.15 | 26698.14 | 37532.01 |
  | set3::bearing_3 | 3 | 2004-03-04 14:12:46 | 64100.15 | 26627.23 | 37472.92 |
  | set3::bearing_3 | 3 | 2004-03-04 11:02:46 | 64290.15 | 27036.56 | 37253.59 |
  | set3::bearing_3 | 3 | 2004-03-04 13:32:46 | 64140.15 | 26905.66 | 37234.49 |
  | set3::bearing_3 | 3 | 2004-03-04 11:12:46 | 64280.15 | 27232.73 | 37047.42 |
  | set3::bearing_3 | 3 | 2004-03-04 23:22:46 | 63550.15 | 26528.94 | 37021.21 |
  | set3::bearing_3 | 3 | 2004-03-05 05:22:46 | 63190.15 | 26240.10 | 36950.05 |
  | set3::bearing_3 | 3 | 2004-03-04 10:42:46 | 64310.15 | 27429.09 | 36881.06 |

---

_Generated by Phase 6 pipeline. Do not edit manually._