# LSTM Results

## Validation Comparison

The following results compare all tested LSTM architecture and hyperparameter configurations on the validation set.

| Configuration             | Epochs | Best Val Loss |  24h RMSE |   24h MAE |    24h R² |  48h RMSE |   48h MAE |    48h R² |  72h RMSE |   72h MAE |    72h R² |
| ------------------------- | -----: | ------------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| `baseline_64_32`          |     13 |      1201.717 |     15.98 |     11.73 |     0.350 |     18.01 |     13.12 |     0.181 |     19.27 |     14.18 |     0.093 |
| **`wider_128_64`**        | **12** |  **1132.268** | **15.35** | **11.60** | **0.400** | **17.45** | **13.10** | **0.232** | **18.82** | **13.97** | **0.135** |
| `deeper_64_64_32`         |     16 |      1413.922 |     18.76 |     13.44 |     0.104 |     19.18 |     13.78 |     0.072 |     20.34 |     14.23 |    -0.010 |
| `bidirectional_64_32`     |     21 |      1282.286 |     15.95 |     11.86 |     0.352 |     18.65 |     13.37 |     0.122 |     20.17 |     14.69 |     0.007 |
| `regularized_96_48_lowlr` |     14 |      1386.363 |     18.18 |     13.19 |     0.159 |     19.35 |     14.00 |     0.054 |     20.11 |     14.54 |     0.013 |

### Selected Configuration

**`wider_128_64`** was selected based on validation performance.

It achieved the lowest validation loss and consistently outperformed the other configurations across all three forecasting horizons:

* **24h:** RMSE = 15.35, MAE = 11.60, R² = 0.400
* **48h:** RMSE = 17.45, MAE = 13.10, R² = 0.232
* **72h:** RMSE = 18.82, MAE = 13.97, R² = 0.135

---

## LSTM Holdout Results

The final evaluation was performed on a **held-out test set that was never used during model training, validation-based model selection, or early stopping**.

The selected `wider_128_64` configuration achieved the following results:

| Forecast Horizon |      RMSE |      MAE |         R² |
| ---------------- | --------: | -------: | ---------: |
| **24h**          | **11.22** | **7.83** |  **0.376** |
| **48h**          | **12.72** | **9.16** |  **0.141** |
| **72h**          | **13.24** | **9.66** | **-0.009** |

### Holdout Performance Summary

The model performs best at the **24-hour forecasting horizon**, with an R² of **0.376** and an RMSE of **11.22**. Performance decreases as the forecasting horizon increases, which is expected for longer-term time-series forecasting.

At 72 hours, the R² is approximately zero (`-0.009`), indicating that the model provides limited predictive value beyond the information captured by the baseline variance at this horizon.

Importantly, these holdout results were obtained **after selecting the architecture using validation data**, providing an unbiased estimate of the selected model's generalization performance.

---

## Final Model

| Property                      | Value                  |
| ----------------------------- | ---------------------- |
| Selected Architecture         | `wider_128_64`         |
| Model Type                    | LSTM                   |
| Selection Criterion           | Validation performance |
| Early Stopping                | Validation loss        |
| Holdout Used During Selection | **No**                 |
| Best Validation Loss          | **1132.268**           |
| Training Epochs               | **12**                 |

### Conclusion

The `wider_128_64` LSTM configuration was selected as the final model because it demonstrated the strongest validation performance across all forecasting horizons. The independent holdout evaluation confirms strong short-term forecasting performance, particularly for the 24-hour horizon, while highlighting the increasing uncertainty associated with longer forecasting horizons.
