# Recognising Physical Activities for Unseen People
*Recognising everyday activities from a waist-worn phone, for people the model has never seen.*

**ES 335 course project (IIT Gandhinagar)**
- Bhavya Chaudhary (25110075): Person A
- Lakshika Aswal (25110179): Person B

**Website (the full write-up, with interactive results):** https://chaudharybhavya.github.io/ML_HAR/


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

To rebuild the website's data files from the saved results (run from the repository root):
```
python docs/make_data.py          # docs/data.js  (cross-validation, final test, errors, leakage splits, physics baseline)
python docs/make_phone_data.py    # docs/phone.js (phone test)
```

`config.py` holds the shared settings: `SEED = 42` and the data path.

**Slow cells:** the lasso cells in `04_logistic.ipynb` use the `saga` solver and take about 25 minutes on a laptop CPU (the laptop will get warm; that is normal). Everything else takes seconds to a few minutes. No GPU is needed.

**Versions:** `04_logistic.ipynb` was run with scikit-learn 1.8; the other notebooks with 1.9.1 (pinned in `requirements.txt`).

## Repository layout
| File | What it does | Owner | Status |
|---|---|---|---|
| `01_signals.ipynb` | Raw signals, physics (gravity vs motion), data-quality checks | A | done |
| `02_features_pca.ipynb` | The 561 features, correlation, PCA | B | done |
| `03_evaluation.ipynb` | Person-wise folds, majority and physics baselines, leakage experiment (random / neighbours removed / person) | A | done |
| `04_logistic.ipynb` | Logistic regression: how it predicts, tuning C, ridge vs lasso | A | done |
| `05_svm.ipynb` | RBF SVM, tuning C and gamma | B | done |
| `06_random_forest.ipynb` | Random forest, feature importance | B | done |
| `07_selection_final_test.ipynb` | Choosing the final model, one final test on the 9 test people, bootstrap interval | A | done |
| `08_errors.ipynb` | Confusion matrix, per-person errors, gravity check, confidence | A | done |
| `09_phone_pipeline.ipynb` | Turning our own phone recordings into windows and features | B | done |
| `10_phone_test.ipynb` | Testing the final model on our own phone recordings | B | done |
| `results/folds.csv` | The saved person → fold table (used by every model) | A | done |
| `results/extra_experiments.csv` | Fold scores (macro-F1, accuracy) for the three leakage splits and the physics baseline (Part 3, Cell 6) | A | done |
| `results/cv_logistic.csv` | Fold scores for all logistic regression settings (63 rows) | A | done |
| `results/cv_svm.csv` | Fold scores for all SVM settings | B | done |
| `results/cv_forest.csv` | Fold scores for all random forest settings | B | done |
| `results/test_predictions_lasso.csv` | Final model's probabilities and predictions for every test window | A | done |
| `results/test_per_person_lasso.csv` | Test macro-F1 for each of the 9 test people | A | done |
| `data/phone/bhavya/` | Our phone recordings of all 6 activities (see below) | A | done |
| `results/phone_features_bhavya.csv` | Phone windows (270) with gravity and the 451 reproduced features, on UCI's scale | B | done |
| `results/phone_predictions_bhavya.csv` | Final model's probabilities and predictions for every phone window | B | done |
| `docs/` | The interactive website, live at https://chaudharybhavya.github.io/ML_HAR/ (GitHub Pages; data rebuilt from `results/` by `docs/make_data.py` and `docs/make_phone_data.py`) | both | done |

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

## Results (cross-validation on 21 training people, 7 person-wise folds; final test below the table)
| Model | Macro-F1 (mean ± std over folds) |
|---|---|
| Majority class (always "lying") | 0.05 (accuracy 0.19) |
| Logistic regression, **random window split** (wrong, leaky) | 0.985 ± 0.007 |
| Physics baseline: logistic regression on 4 numbers (mean gravity x, y, z; motion), person split | 0.821 ± 0.069 |
| Logistic regression, random split with overlapping neighbours removed (people still shared) | 0.977 ± 0.006 |
| Logistic regression, **person split** (correct), default C = 1 | 0.933 ± 0.036 |
| Logistic regression, ridge, C = 0.01 (one-SE choice) | 0.928 ± 0.039 |
| **Logistic regression, lasso, C = 0.1 (one-SE choice, ~180 of 561 features)** | **0.934 ± 0.036** |
| RBF SVM, C = 1, gamma = 0.001 (one-SE choice) | 0.931 ± 0.027 |
| Random forest, 100 trees, max_depth = 10, max_features = "sqrt" (one-SE choice) | 0.907 ± 0.041 |

**Final test (Part 7):** lasso logistic regression, C = 0.1, trained on all 21 training people and tested once on the 9 test people: **macro-F1 0.945** (accuracy 0.946), **95% bootstrap interval over people 0.906–0.975**.

The random split hides most of the drop on new people, because neighbouring windows overlap by 50% (95.5% of same-bout pairs share readings exactly) and the same people appear on both sides. Part 3, Cell 6 separates the two causes by removing overlapping neighbours from training: the score only falls from 0.985 to 0.977, so about 85% of the inflation comes from seeing the same people, not from the overlap (an upper bound on the overlap's share, since removing neighbours also removes about a quarter of the training windows). The 4-number physics baseline (0.821) shows the 561 engineered features add about 0.11 macro-F1 for new people.

All ± values are the sample standard deviation over the 7 folds; the standard error used for choosing is that ÷ √7.

These are cross-validation scores used for *choosing*. They are slightly optimistic. The honest number is the single final test on the 9 test people (above).

## What Part 4 found (logistic regression)
- The model gives each activity a linear score from the 561 features; softmax turns the 6 scores into probabilities. We reproduced its output by hand.
- Regularisation strength (C) barely matters: every C from 0.01 to 100 scores within noise of the best. Differences *between people* (fold scores 0.885–0.981) are much bigger than differences between settings.
- Lasso reaches the same score with about a third of the features, but the features it keeps change from fold to fold (many features are near-copies), so we do not treat its choices as "the important features".
- The model's confidence is informative but not fully trustworthy: about 1 in 5 of its mistakes happen at more than 90% confidence.
- **Candidate carried to Part 7: lasso, C = 0.1.**

## What Part 2 found (the features and PCA)
- The 561 features are families of a few ideas (movement size, rhythm/frequency, orientation), repeated over 3 axes, 2 sensors, and time vs frequency. Only 53 describe gravity (orientation).
- They overlap heavily: 8,206 pairs with |r| > 0.9 (207 with |r| > 0.99), and 431 features have a near-copy. Some are exact copies (e.g. the `tGravityAccMag` columns equal the `tBodyAccMag` ones).
- PCA (scaler + PCA in a pipeline): 63 / 102 / 179 components keep 90 / 95 / 99% of the variance. PC1 (51%) is movement intensity and almost separates moving from still; lying only separates on PC4/PC5 (gravity); sitting vs standing and the three walking types overlap.
- Logistic regression on 100 components scores 0.920 vs 0.933 with all 561 features, so we do not use PCA in front of the models: it keeps the largest variation, not the most useful.

## What Parts 5 and 6 found (SVM, random forest)
- **SVM:** gamma matters far more than C. Too large (0.1) and the model collapses to 0.06 (every window looks unlike every other). 11 of 25 settings are within one SE of the best (C = 10, gamma = 0.001: 0.938); the simplest is C = 1, gamma = 0.001 (0.931). It ties with lasso (better on 2 of 7 folds) and varies a little less between people.
- **Random forest:** the settings barely matter (all 27 between 0.891 and 0.909); fewer features per split is slightly better. The simplest within one SE is 100 trees, depth 10, "sqrt" (0.907), about 0.025 below the other two models, and worst on fold 1 (people 3, 8, 21: 0.838).
- **Importance:** impurity and permutation importance disagree almost completely (1 shared feature in their top 15), and shuffling any single feature costs at most 0.0014 macro-F1. Shuffling one gravity feature together with its 4 near-copies costs 0.196, so importance belongs to groups of correlated features, the same reason lasso's choices are unstable.
- **Candidates carried to Part 7:** SVM C = 1, gamma = 0.001; random forest 100 trees, max_depth = 10, max_features = "sqrt".

## What Part 7 found (choosing the model, final test)
- **Choosing:** the same one-SE rule across the three candidates gives a threshold of 0.934 − 0.014 = 0.920. Lasso (0.934) and SVM (0.931) qualify; the forest (0.907) does not. Lasso vs SVM fold by fold: SVM is better on only 2 of 7 folds (differences −0.018 to +0.022), so they cannot be told apart. We chose the **simpler** one: **lasso logistic regression, C = 0.1**. The decision was written down before the test set was opened.
- "Hard people" depend partly on the model: fold 2 (people 14, 26, 29) is the linear models' worst fold but the forest's best.
- **Final test** (trained on all 21 training people, evaluated once on 9 test people, 2,947 windows): **macro-F1 0.945, accuracy 0.946.** This is close to the cross-validation estimate of 0.934.
- **Uncertainty:** 95% bootstrap interval **0.906–0.975**, resampling the 9 *people* (not windows) 2,000 times, because a person's windows are not independent.
- **Per test person:** 24: 1.000 · 13: 0.990 · 2: 0.983 · 12: 0.969 · 18: 0.959 · 4: 0.954 · 20: 0.895 · 9: 0.881 · 10: 0.826.
- For context only (run after the decision, not used for choosing): SVM test 0.947, forest 0.919.
- Leave-one-person-out was not run: the 7-fold out-of-fold predictions already score every training person, and 21 more slow lasso fits would add little.

## What Part 8 found (errors on the test people)
- **160 errors in two blocks:** 74 among the still postures (sit → stand 55, stand → sit 19) and 82 among the walking types (up → walk 36, down → up 33). Only 4 errors between moving and still. Lying is perfect (537 of 537).
- **Errors are concentrated:** people 10, 9 and 20 make 113 of the 160 errors (71%); person 24 makes none. Stairs errors are person-specific (down → up comes only from people 20 and 18); sit → stand appears in 7 of 9 people, so it is a weakness of the method.
- **Physics check (gravity tilt):** most sit/stand errors lie where the sitting and standing tilt clouds overlap. Some sitting windows of people 9 and 10 lie outside every other person's cloud: a posture the model never saw, which it confidently calls "standing". Errors come in clumps (whole bouts).
- **Confidence:** below 0.6 confidence the model is right 58% of the time; at 0.9 or above, 99%. Answering only above 0.9 would cover 78% of windows and avoid about 86% of errors. The 23 errors made with more than 90% confidence are mostly person 10's unfamiliar sitting posture.

## Limitations
1. **Who the model learned from:** 21 adults doing scripted activities in a lab, one phone model at one position. Other placements, phones, ages and free-living behaviour are untested (apart from our own recordings).
2. **Unfamiliar postures fail confidently:** the model only knows postures like its training people's.
3. **Few people:** 9 test people give a wide interval (0.906–0.975); individuals range from 0.83 to 1.00.
4. **Correlated errors:** windows within a bout and a person are not independent evidence.
5. **One window at a time:** smoothing predictions over consecutive windows is not part of the final model; it was only tried as context on our phone recordings (Part 10: a vote over the last 5 windows raised macro-F1 from 0.860 to 0.886).
6. **No transitions** (e.g. standing up) in the six classes.
7. **Prediction, not explanation:** feature choices and importances are unstable among correlated features, so we make no causal claims.

## Our phone recordings (Parts 9–10)
- **Pipeline (Part 9):** resample to 50 Hz, convert to g and rad/s, align the axes if needed (not needed for our recordings), separate gravity with a 0.3 Hz filter, cut 2.56 s windows with 50% overlap, compute the features. Checked on UCI's own training signals, it reproduces 451 of the 561 features; logistic regression on those 451 scores 0.931 (vs 0.933 on all 561).
- **Recordings (person `bhavya`, 14 Nov 2026):** vivo V2439 (Android 16), phyphox (accelerometer *with* g + gyroscope) at 50 Hz. Phone at the waist on the side of the hip, landscape, screen facing outward, same position for every recording. About 67 s per activity (lying 83 s), one file per activity, about 43 windows per activity after the loader cuts the first and last 5 s.
- **Orientation check:** standing gives gravity x = +1.00 g, y = 0.01, z = 0.17 (average of UCI's 21 training people: 1.00, −0.16, −0.03), so no axis remapping is needed. Sitting gives z = 0.34 (UCI average: 0.15).
- **Notes:** the upstairs file includes a short landing/turn (around 35–40 s); two extra, shorter stair recordings were made but not used (the loader reads one file per activity). Lying puts gravity on y (0.99) rather than split between y and z as in UCI (average 0.65, 0.56), because the phone sat more on the side of the hip.
- **Expected confusions** from simple summaries: walking vs upstairs (same bounce, differ by forward tilt and rhythm) and sitting vs standing (differ only by tilt).
- **Results (Part 10):** a blind test; the final model was fixed in Part 7 before the recordings were made, and nothing was tuned on them.
  - The 561-feature model cannot take phone windows, so the **same setting** (lasso, C = 0.1) was trained on the **451 reproduced features** of the 21 training people. Cross-validation: 0.926 ± 0.042 (vs 0.934 ± 0.036 with all 561), lower on every fold, but the average drop (0.008) is smaller than one SE (0.014).
  - **Phone: macro-F1 0.860, accuracy 0.870** (235 of 270 windows). Recall: walking 1.000, lying 1.000, downstairs 0.953, standing 0.860, sitting 0.791, upstairs 0.581.
  - **Same two error blocks as Part 8:** walking types 19 of 35 errors (up → walk 13) and sitting vs standing 14; moving vs still only 2.
  - **Why sitting vs standing fails:** UCI's people separate the two mostly by gravity y (averages: sitting +0.14 g, standing −0.16 g). On this phone both postures sit at y ≈ 0 and differ only along z (0.34 vs 0.17 g), where UCI's sitting and standing overlap.
  - **Confidence:** at 0.9 or above the model is still right 98.7% of the time (Part 8: 99%), but only 56% of phone windows reach that confidence (Part 8: 78%).
  - Context only: a majority vote over the last 5 windows raises macro-F1 to 0.886 (not part of the final model).
  - **Caveat:** one person, one phone, one session, so no interval; it shows that the model transfers and how it fails, not how well it works for people in general.

## Key data facts (from Parts 1 and 3)
- No missing values, no duplicate rows; all features are scaled to [-1, 1].
- Classes are fairly balanced (13.4–19.1% of the training windows each); 281–409 windows per person.
- Gravity direction separates lying clearly; sitting vs standing overlap; the three walking types overlap.
- Folds: 7 folds × 3 people each; no person appears on both sides.

## Status
All ten notebooks, the results files and the website are complete. The website presents the full reasoning (question, data, preprocessing, baseline and models, evaluation, results and error analysis, limitations, conclusions); this repository holds the code, the exact split (`results/folds.csv`) and every saved result needed to check it.

## AI assistance (disclosure)
We used Claude (Anthropic) as a tutor and assistant. It explained concepts and the physics of the data; helped with debugging; suggested and wrote notebook code under our direction; drafted parts of the notebook text, the README and the website; and helped build the structure and interactive charts of the website from our saved results.

We planned the analyses, set the rules they follow (person-wise folds, the one-standard-error rule, a blind phone test with nothing tuned on it) and made the decisions. We checked every reported number against our notebook outputs, we are responsible for all results, and we can explain every step.

## References and citations
- **Dataset:** Anguita, D., Ghio, A., Oneto, L., Parra, X. and Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN 2013*. UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones (CC BY 4.0). The descriptions of the authors' preprocessing (noise filter, 0.3 Hz gravity filter, 2.56 s windows with 50% overlap, feature definitions) come from the dataset's own `README.txt` and `features_info.txt`.
- **Software:** Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR*, 12, 2825–2830. Also NumPy, pandas, SciPy and matplotlib (versions in `requirements.txt`).
- **Phone recording app:** Staacks, S., Hütz, S., Heinke, H. and Stampfer, C. (2018). Advanced tools for smartphone-based experiments: phyphox. *Physics Education*, 53(4), 045009.
- **Evaluation design:** Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C. and Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), 1–9. https://doi.org/10.1093/gigascience/gix019 (record-wise vs subject-wise cross-validation).
- **Methods:**
  - Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS B*, 58(1), 267–288.
  - Cortes, C. and Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
  - Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
  - Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer (cross-validation and the one-standard-error rule, section 7.10).
  - Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- **Course material:** ES 335 lecture notes (logistic regression, regularisation, metrics, SVM, trees and forests).
- **External code:** none copied from other repositories or tutorials. How AI tools were used is in the disclosure above.
