# Skills demonstrated in this project

Each skill is tied to where it appears in the notebook (step numbers) or the report.

## Machine learning and deep learning
| Skill | Evidence |
|---|---|
| Generative adversarial networks | Full TimeGAN with five networks: embedder, recovery, generator, supervisor, discriminator (Step 8) |
| Recurrent networks (GRU, LSTM) | Stacked and bidirectional GRUs for TimeGAN; LSTM classifier and forecaster for evaluation (Steps 8, 13, 14) |
| Custom training loops in TensorFlow 2 / Keras | `tf.function` and `GradientTape` steps for each training phase, separate optimisers, and a discriminator update gate at loss > 0.15 (Step 9) |
| Multi-phase and multi-objective training | Reconstruction, supervised, adversarial and moment-matching losses with weighted combinations (Step 9) |
| Reading and re-implementing research papers | TensorFlow 2 port of the NeurIPS 2019 official implementation, with settings traced to the paper |
| Training diagnostics | Loss curves for all three phases, check for divergence or collapse (Step 9, report §5.2) |

## Time-series data engineering
| Skill | Evidence |
|---|---|
| Large data handling | 2.07M per-minute rows reduced to 34,589 hourly points (Step 4) |
| Cleaning and imputation | Datetime parsing, de-duplication, `?` as NaN, outage analysis, time-based interpolation (Steps 3-4) |
| Leakage-aware splitting and scaling | Chronological 80/20 split, MinMaxScaler fitted on train only, test clipped to [0, 1] (Step 6) |
| Sliding-window sequence construction | Vectorised windowing with `sliding_window_view`, shape (N, 24, 7) (Step 7) |
| Exploratory analysis | Daily cycle, seasonality, correlation structure (Step 5) |

## Evaluation and statistics
| Skill | Evidence |
|---|---|
| Evaluating synthetic data on several axes | Visual, distributional, PCA and t-SNE, correlation gap, autocorrelation gap, discriminative score, TSTR predictive score (Steps 11-14) |
| Train-on-Synthetic-Test-on-Real (TSTR) | Forecaster trained on synthetic vs. real data, tested on real held-out data (Step 14) |
| Sensible baselines | Persistence forecast as a naive reference (Step 14) |
| Reproducibility | Fixed seeds, central config cell, 3-seed repeats reported as mean ± std |
| Dimensionality reduction | PCA fitted on real data only, t-SNE on pooled samples (Step 11) |
| Time-series statistics | Autocorrelation at lags 1-23, correlation-matrix comparison (Step 12) |

## Scientific communication
| Skill | Evidence |
|---|---|
| Honest reporting of negative results | 99.2% classifier accuracy, TSTR worse than persistence, and the reasons are all stated plainly |
| Critical analysis | Separates what worked from what did not, flags test-period drift as a confound, marks untested explanations as untested |
| Limitations and future work | Eight limitations and concrete next steps (report §8-9) |
| Technical writing | Structured report with equations, tables and figures; step-by-step notebook with a "what and why" for each cell |
| Documentation | Every notebook step explains purpose, reasoning and what to look for |

## Tools and libraries
Python, TensorFlow 2 / Keras, NumPy, pandas, scikit-learn (PCA, t-SNE, MinMaxScaler, train_test_split), Matplotlib, Seaborn, Google Colab (GPU), kagglehub, Git and GitHub, LaTeX (report).

## Resume and LinkedIn bullets

- Implemented **TimeGAN** in TensorFlow 2 (five GRU networks, three-phase training with custom `GradientTape` loops) to generate synthetic household electricity load profiles from 2.07M per-minute readings.
- Built an end-to-end time-series pipeline: cleaning, hourly resampling, interpolation, leakage-safe chronological split, and sliding-window construction.
- Designed a multi-metric evaluation suite (PCA/t-SNE, correlation and autocorrelation gaps, LSTM discriminative score, TSTR forecasting) averaged over 3 seeds.
- Quantified the fidelity gap rather than overstating results: 99.2% real/synthetic classifier accuracy, TSTR/TRTR MAE ratio of 1.16 (0.398 vs 0.343 kW), with analysis of causes and a roadmap for improvement.
- Authored a full technical report and a documented, reproducible notebook, including limitations, drift confounds and future work.
