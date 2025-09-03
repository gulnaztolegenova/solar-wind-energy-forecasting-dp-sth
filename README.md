# **solar-wind-energy-forecasting-dp-sth** ⚡️🌬️📊

**Dual-Purpose Spatio-Temporal Hybrid (DP-STH) for hourly solar & wind power forecasting and extreme-event detection in smart-city grids.**
*This repository is a research-grade, end-to-end workflow: curated data preparation, seasonality diagnostics, model training (regression + extremes), strict temporal validation, and publication-quality visualizations.*

---

## 🔎 What this repository is about (quick tour)

* **Problem.** Short-term (hourly) **solar** and **wind** generation forecasting with **concurrent detection of extreme events** relevant for grid operations in a *smart-city* setting.
* **Data.** Public, hourly **PV + wind** dataset (Mendeley, DOI below). We **merge** the originally split files into **one time-aligned table**, enrich with **calendar features**, and keep **train-only** statistics for fair evaluation.
* **Methods.** A **family of models trained separately** on the same, unified pipeline: LSTM/GRU, LSTM+GRU, ST-Transformer, and the **proposed DP-STH** (parallel **causal TCN + light causal Transformer**), each with a **multi-task head**:

  * **Regression** (unweighted/uncertainty-weighted MSE) for power,
  * **Extreme classification** (Focal/BCE) at the **95th-percentile** threshold computed **only on TRAIN**.
* **Validation.** **TimeSeriesSplit** cross-validation **and** **80/20 holdout**; window length **L = 24 h** justified by **autocorrelation** analysis (dominant diurnal cycle).
* **Outputs.** Clean notebooks, figure set (EDA, seasonality, architecture diagrams, evaluation windows), and tables ready for manuscript submission.

---

## 🔗 Data access

**Primary source (external):**
**Wind and Solar Power Generation Dataset** — *Mendeley Data* (Version 1, 10 Oct 2024)
**DOI:** `10.17632/gxc6j5btrx.1` — Contributor: **Yue Liu**
➡️ [https://data.mendeley.com/datasets/gxc6j5btrx/1](https://data.mendeley.com/datasets/gxc6j5btrx/1)

**What we do with the data**

* Merge PV and wind files into a **single** hourly table.
* **Sort & deduplicate** timestamps; drop near-constant columns.
* **Add calendar**: hour, day, month, day-of-year, week, season.
* Produce **targets**: `kWh_solar_power_solar`, `kWh_wind_power_wind`.
* Produce **predictors** (examples): `surface_irradiance_solar`, `toa_irradiance_solar`, `temperature_solar`, `humidity_solar`, `wind_speed_wind`, `air_density_wind`, `hour_index_*`, `season_derived`, etc.

---

## 🧰 Pipeline at a glance

**Train-only preprocessing**

* Two-sided **winsorization** (q01/q99) per feature
* **Median imputation**
* **Signed-log1p** variance stabilization
* **Min-Max scaling** (fit on **TRAIN** only)
* Build inputs:

  * **`X_seq`**: last **L = 24** steps (left-pad if needed)
  * **`X_flat`**: last step **(i)**

**Extreme labels**

* `extreme = 1` if target > **95th percentile** (computed on **TRAIN**), applied to val/test **without refitting**.

**Seasonality & window selection**

* Manual ACF (lags 1–48; 24/48/168 h) → **24 h dominates**, so **L = 24 h** balances information and model size.

---

## 🧪 Models (each trained **separately**, not an ensemble)

* **MTL\_LSTM**, **MTL\_GRU** — sequence encoders + shared multi-task head
* **MTL\_LSTM\_GRU** — parallel LSTM and GRU streams
* **MTL\_LSTM\_GRU\_STT** — adds a **spatio-temporal Transformer** branch
* **STT (stand-alone)** — causal Transformer with time positional encoding
* **🆕 DP-STH (proposed)** — **Dual-Purpose Spatio-Temporal Hybrid**

  * **Encoders in parallel:**

    * **Causal TCN** with dilated residual blocks (local bursts, robustness)
    * **Light causal Transformer** (longer dependencies at lower cost)
  * Optional LSTM/GRU traces + a **dense last-step** path
  * Concatenate → Dense → Dropout → **multi-task head**

    * **Power regression** (UW-MSE)
    * **Extreme detection** (Focal/BCE)

**Training & metrics**

* Adam/AdamW, early stopping, LR scheduling
* **RMSE, MAE, sMAPE/MAPE, R², EVS, ROC-AUC** (extremes)
* **TimeSeriesSplit** CV + **80/20 holdout**

---

## 📊 What you can reproduce

* **EDA & feature relevance:** distributions, correlation matrix, **Mutual Information**, **SHAP** (global bars + grouped view).
* **Seasonality visuals:** hourly & monthly profiles; **Month×Hour heatmaps**; ACF bar plots and **window comparison figure**.
* **Architecture diagrams:** data pipeline; MTL variants; stand-alone ST-Transformer; **DP-STH**.
* **Evaluation plots:** publication-grade 1- and 2-week windows (ground truth vs forecast) and **P(extreme)** timelines with thresholds.
* **Metrics tables:** **CV (mean ± std; CI90)** and **Holdout** summaries for Solar and Wind (**no numbers fixed in README**; regenerate in notebooks).

---

## 🗂️ Repository layout

```
.
├── data_analysis.ipynb                 # EDA, MI/SHAP, seasonality & ACF
├── baseline_Transformer_model.ipynb    # Stand-alone spatio-temporal Transformer
├── new_hybrid_model.ipynb              # DP-STH training & evaluation
├── renewables_combined_FULL*.csv       # (optional) merged time-aligned dataset
└── README.md
```

> If you use the external raw files, place them under `data/raw/` and follow the merge/clean steps in the EDA notebook.

---

## 🚀 Getting started

```bash
# Clone
git clone https://github.com/gulnaztolegenova/solar-wind-energy-forecasting-dp-sth.git
cd solar-wind-energy-forecasting-dp-sth

# Environment
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn shap tensorflow==2.*

# Run notebooks (recommended order)
jupyter lab
# 1) data_analysis.ipynb
# 2) baseline_Transformer_model.ipynb
# 3) new_hybrid_model.ipynb
```

**Repro tips**

* Keep `shuffle=False` for all temporal splits.
* Fit **winsorization, imputers, scalers, and extreme thresholds on TRAIN** only.
* Save model with the **scaler**, **window length L**, **horizon H**, and **extreme threshold**.

---

## 🧠 Why DP-STH (novelty in brief)

* **Dual-purpose learning** (point accuracy **and** risk signaling) in one head **without** conflating tasks—suited to operational decision-making in smart grids.
* **Parallel causal encoders** (TCN + light Transformer) combine **local burst sensitivity** and **long-range context** with **lower complexity** than heavy attention-only stacks.
* **Methodological hygiene**: unified, **train-only** preprocessing and **fixed extreme thresholds** ensure **fair comparability** across PV & wind and **robust generalization**.

---

## 📜 License & citation

Released under the **MIT License** (add `LICENSE` if not present).
Please cite both this repository and the dataset if you use the code or figures.

**Repository**

```bibtex
@misc{dp_sth_solar_wind_2025,
  title  = {DP-STH: Dual-Purpose Spatio-Temporal Hybrid for Solar and Wind Power Forecasting},
  author = {Tulegenova Gulnaz},
  year   = {2025},
  url    = {https://github.com/gulnaztolegenova/solar-wind-energy-forecasting-dp-sth}
}
```

**Dataset**

```bibtex
@dataset{liu_2024_mendeley,
  title  = {Wind and Solar Power Generation Dataset},
  author = {Yue Liu},
  year   = {2024},
  doi    = {10.17632/gxc6j5btrx.1},
  url    = {https://data.mendeley.com/datasets/gxc6j5btrx/1}
}
```

---
