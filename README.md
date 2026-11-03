# ROGII Wellbore Geology Prediction

Predicting **True Vertical Thickness (TVT)** along the unseen part of horizontal wells from gamma-ray (GR) logs and offset-well information. Built for the Kaggle competition *ROGII Wellbore Geology Prediction*.

> **Status:** complete pipeline, awaiting its first full run on the competition data. Validation results will be added here after that run.

## Problem

Horizontal wells are steered through a thin target formation. Past a known "prediction start" (PS) point, the TVT position of the well relative to the geology is hidden, but the gamma-ray log keeps being recorded. The task is to predict TVT for every hidden row of each test well, using:

- the well's own trajectory (`MD`, `X`, `Y`, `Z`) and gamma-ray (`GR`) log,
- the known TVT before PS (`TVT_input`),
- a **typewell**: a reference vertical log of GR against TVT for the area.

Submissions are scored on prediction error in feet (RMSE).

## Approach

Five independent estimators, each exploiting a different signal, are blended by cross-validated accuracy.

| Method | Idea |
|---|---|
| **Contact model** (`contact`) | Physical model: derive TVT from geological contact depths, auto-selecting the formation that minimises residual on the known (pre-PS) section. |
| **Particle filter** (`pf`) | Tracks TVT through the hidden zone with a particle cloud (momentum + noise on dip rate). Particles are weighted by how well the typewell GR at the proposed TVT matches the observed horizontal GR, with resampling when effective sample size drops. Run as a multi-seed, likelihood-weighted ensemble. |
| **Beam search** (`beam`) | Searches for the lowest-cost TVT path that makes the horizontal GR match the typewell GR, with smoothness penalties on TVT moves. Averaged over 14 beam configurations (width, move cost, GR error scale, smoothing radius). |
| **GR k-NN** (`knn`) | Uses windows of pre-PS GR as templates; each hidden GR window is matched to its nearest known windows and TVT is a distance-weighted average of their TVT. |
| **Offset-well prior** (`offset`) | Extrapolates from the last known TVT using the median TVT-per-MD rate of similar neighbouring wells (found via a well-signature search). |

**Ensembling.** Each method is scored on a set of calibration wells (up to 8 training wells spread across the set) by RMSE on their hidden zone. Weights are a softmax over negative RMSE (temperature 0.5), so better methods get more weight. Final inference uses a heavier particle filter (500 particles, 128 seeds) than calibration (250 particles, 16 seeds).

**Feature engineering.** Rolling GR statistics (mean, std, min, max, median), first and second differences, TVT-input rolling features, azimuth and inclination per step, and turn rate.

## Evaluation

The last cell of the notebook evaluates every method on training wells whose hidden-zone TVT is known, and prints and saves (`cv_results.csv`) the mean and spread of RMSE per well for:

- **identity baseline:** the last known TVT held constant (any method must beat this to be useful),
- each of the five methods,
- the ensemble with weights tuned on the same wells (slightly optimistic), and
- the ensemble with **leave-one-well-out** weights (the honest estimate to quote).

The offset-well prior excludes the well being predicted from its own neighbours, so no well can see its own answer.

## Repository layout

```
notebooks/
  rogii_wellbore_geology.ipynb    main pipeline
  experiments/                    earlier iterations (exploratory; their saved outputs
                                  come from generated placeholder data, not competition data)
requirements.txt
```

## Running it

1. Download the competition data from Kaggle and place it so that `train/`, `test/` and `sample_submission.csv` sit in one folder.
2. Set `DATA_CANDIDATES` in the configuration cell to that folder. The notebook also auto-discovers `/kaggle/input/**` when run on Kaggle.
3. `pip install -r requirements.txt`
4. Run the notebook top to bottom. It writes `submission.csv` and checks row count, unique IDs and absence of NaNs.

Random seed is fixed (`SEED = 42`).

## Data

Competition data is not redistributed here. See the competition page for terms.

## Possible next steps

- Replace hand-tuned weights with a learned stacker.
- Add uncertainty estimates from the particle spread.
- Evaluate on more calibration wells to reduce variance in the ensemble weights.
