# Recognising everyday activities from a waist-worn phone

**ES 335 course project (IIT Gandhinagar)**
- Bhavya Chaudhary (25110075): Person A
- Lakshika Aswal (25110179): Person B

**Final deadline:** 17 November 2026: website + this repository.

## The question
Given 2.56 seconds of accelerometer and gyroscope readings from a phone worn at the waist, can we tell which of six activities a **new person** (never seen in training) is doing?

The six activities: walking, walking upstairs, walking downstairs, sitting, standing, lying.

- **One row** = one 2.56 s window (128 readings at 50 Hz) from one person.
- **Target** = the activity label of that window.
- **Prediction time:** at the end of each window, using only that window's readings.
- **Unit of generalisation** = the person. All evaluation is done on people the model has never seen.

## Data
- **UCI HAR dataset** (Anguita et al., 2013), Samsung Galaxy S II on the waist, 30 volunteers.
- 7,352 training windows (21 people) + 2,947 test windows (9 people).
- 561 engineered features per window, plus the raw signals (9 channels).
- **Download:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones (licence CC BY 4.0)
- The data is **not** in this repo. Unzip it so this path exists: `data/raw/UCI HAR Dataset/`

## How to run
```
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Then open the notebooks in order in VS Code (Jupyter), and run each with **Restart & Run All**.

`config.py` holds the shared settings: `SEED = 42` and the data path.

**Slow cells:** the lasso cells in `04_logistic.ipynb` use the `saga` solver and take about 25 minutes on a laptop CPU (the laptop will get warm; that is normal). Everything else takes seconds to a few minutes. No GPU is needed.

## Repository layout
| File | What it does | Owner | Status |
|---|---|---|---|
| `01_signals.ipynb` | Raw signals, physics (gravity vs motion), data-quality checks | A | done |
| `02_features_pca.ipynb` | The 561 features, correlation, PCA | B | to do |
| `03_evaluation.ipynb` | Person-wise folds, majority baseline, leakage experiment | A | done |
| `04_logistic.ipynb` | Logistic regression: how it predicts, tuning C, ridge vs lasso | A | done |
| `05_svm.ipynb` | RBF SVM, tuning C and gamma | B | to do |
| `06_random_forest.ipynb` | Random forest, feature importance | B | to do |
| `07_selection_final_test.ipynb` | Pick the model, one final test on the 9 test people, uncertainty | A | waiting for B's results (05, 06) |
| `08_errors.ipynb` | Confusion matrix, hardest people, representative errors | A | waiting for 07 |
| `09_phone_pipeline.ipynb` | Turning our own phone recordings into windows and features | B | to do |
| `10_phone_test.ipynb` | Testing the final model on our own phone recordings | B | waiting for 07, 09 |
| `results/folds.csv` | The saved person → fold table (used by every model) | A | done |
| `results/cv_logistic.csv` | Fold scores for all logistic regression settings (63 rows) | A | done |
| `results/cv_svm.csv` | Fold scores for all SVM settings | B | to do |
| `results/cv_forest.csv` | Fold scores for all random forest settings | B | to do |
| `website/` | The interactive site (GitHub Pages) | both | to do |

## Work split
- **Person A (Bhavya):** Parts 1, 3, 4, 7, 8, plus the website sections for them.
- **Person B (Lakshika):** Parts 2, 5, 6, 9, 10, plus the website sections for them.
- **Shared:** Part 11 (website shell), Part 12 (README, AI disclosure, viva prep).

## Rules for both of us (so results are comparable)
1. **Run `git pull` before you start working**, and push when you stop.
2. **Only edit your own notebooks.** Discuss before changing shared files (`config.py`, `README.md`, `requirements.txt`).
3. **Never open the `test/` folder** (the 9 test people) until Part 7. All tuning uses the 21 training people only.
4. **Every model uses the saved folds**, loaded like this:
   ```python
   from config import SEED, DATA
   folds_table = pd.read_csv("results/folds.csv", index_col="person")["fold"]
   fold_of_window = folds_table.loc[groups].values
   saved_folds = PredefinedSplit(fold_of_window)
   # then: cross_val_score(model, X, y, cv=saved_folds, scoring="f1_macro")
   ```
5. **Anything that learns from data goes inside a pipeline** (for example `make_pipeline(StandardScaler(), model)`), so the scaling or PCA is fitted on each fold's training people only.
6. **Main metric: macro-F1.** Use `random_state=SEED` wherever a model has randomness.
7. **Save your fold scores for every setting you try** to `results/cv_svm.csv` and `results/cv_forest.csv`, with the same columns as `results/cv_logistic.csv`: `model, setting, fold, macro_f1` (one row per fold, 7 rows per setting). Example row: `svm_rbf, C=10 gamma=0.001, 3, 0.951`. Part 7 combines these three files to choose the final model, so please also tell A which setting you chose and why.
8. **Choosing a setting within a model:** use the one-standard-error rule. SE = std over the 7 folds ÷ √7; among the settings within one SE of the best mean, pick the simplest one. See `04_logistic.ipynb`, Cell 6.
9. **scikit-learn 1.8 note:** `penalty="l1"` is deprecated in `LogisticRegression`; use `l1_ratio=1` (lasso) or the default `l1_ratio=0` (ridge).

## Results so far (21 training people, 7 person-wise folds)
| Model | Macro-F1 (mean ± std over folds) |
|---|---|
| Majority class (always "lying") | 0.05 (accuracy 0.19) |
| Logistic regression, **random window split** (wrong, leaky) | 0.985 ± 0.006 |
| Logistic regression, **person split** (correct), default C = 1 | 0.933 ± 0.033 |
| Logistic regression, ridge, C = 0.01 (one-SE choice) | 0.928 ± 0.039 |
| **Logistic regression, lasso, C = 0.1 (one-SE choice, ~180 of 561 features)** | **0.934 ± 0.036** |

The random split hides most of the errors on new people (about 1.5% vs 6.7% error), because neighbouring windows overlap by 50% (95.5% of same-bout pairs share readings exactly) and the same people appear on both sides.

These are cross-validation scores used for *choosing*. They are slightly optimistic. The honest number comes from the single final test on the 9 test people (Part 7).

## What Part 4 found (logistic regression)
- The model gives each activity a linear score from the 561 features; softmax turns the 6 scores into probabilities. We reproduced its output by hand.
- Regularisation strength (C) barely matters: every C from 0.01 to 100 scores within noise of the best. Differences *between people* (fold scores 0.885–0.981) are much bigger than differences between settings.
- Lasso reaches the same score with about a third of the features, but the features it keeps change from fold to fold (many features are near-copies), so we do not treat its choices as "the important features".
- The model's confidence is informative but not fully trustworthy: about 1 in 5 of its mistakes happen at more than 90% confidence.
- **Candidate carried to Part 7: lasso, C = 0.1.**

## Key data facts (from Modules 1 and 3)
- No missing values, no duplicate rows; all features are scaled to [-1, 1].
- Classes are fairly balanced (13.7–18.9% each); 281–409 windows per person.
- Gravity direction separates lying clearly; sitting vs standing overlap; the three walking types overlap.
- Folds: 7 folds × 3 people each; no person appears on both sides.

## Next steps
- **A:** blocked until `results/cv_svm.csv` and `results/cv_forest.csv` are pushed. Then Part 7 (model selection, final test, bootstrap over people, leave-one-person-out), then Part 8 (error analysis).
- **B:** `git pull`; then `02_features_pca`, `05_svm`, `06_random_forest`, following rules 4–8. Push the two CSV files as soon as they are ready, and tell A the chosen settings. Then `09_phone_pipeline`.
- **Both:** agree on the website layout (the main interaction is a toggle between the random split and the person split).

## AI assistance (disclosure)
We used Claude (Anthropic) as a tutor. It:
- explained concepts and the physics;
- helped with debugging;
- drafted parts of this README.

We are responsible for all results and can explain every step. 
