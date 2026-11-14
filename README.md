# Learning Fair Representations (Zemel et al. 2013) — study for Assignment 4

Paper: *Learning Fair Representations*, Zemel, Wu, Swersky, Pitassi & Dwork, ICML 2013.
Code: `fairlearn.preprocessing.PrototypeRepresentationLearner` (fairlearn 0.15).
This is the repository linked from the report; the fork at
`https://github.com/RendelGadaffi/fairlearn/tree/a4-fair-representations-study` carries the
same study under `a4_study/` (identical code, data and tables).

LFR is a **pre-processing** method. It learns K prototypes in the input space and maps
every person to a softmax over the negative distances to them. The mapping is trained to
keep reconstruction error low, keep label accuracy high, and make the two protected
groups use the prototypes at the same rate. A classifier then trains on the prototype
coordinates and inherits the fairness of the representation.

This repository has the code, the data, the sweep behind the chosen parameters, and the
results tables below.

## How to reproduce

```bash
git clone https://github.com/RendelGadaffi/iafx-a4-fair-representations.git
cd iafx-a4-fair-representations
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py     # prints the summary, writes results/results.json
python src/sweep.py              # the fairness-weight grid, writes results/sweep.csv
```

The data files are committed under `data/`, so there is nothing to download. Every number
in the report is in `results/sweep.csv`, one row per dataset and fairness weight, so the
tables can be checked without running anything.

Runtimes are in the `seconds` column of `results/sweep.csv`: about 40 s for a German fit,
12 minutes for a COMPAS fit and 54 minutes for an Adult fit, measured with the five weights
of one dataset running side by side. Run them one at a time and they get proportionally
faster. The logistic baselines take seconds.

## 1. Setup

### 1.1 Datasets

| dataset | n | label (positive) | protected attribute (unprivileged) |
|---|---|---|---|
| Adult (UCI) | 48,842 | income > 50K | sex = Female |
| German Credit (UCI) | 1,000 | credit granted | age < 25 |
| COMPAS (ProPublica) | 5,714 | two-year recidivism | sex = Female |

`data/README.md` documents the sources. One note on Adult: the LFR paper cites the
45,222-row collection; the current UCI release has 48,842 rows and no cleaning of it
reproduces that number, so I keep all rows (the rows the old cleaning dropped carry
"?" only in native-country, a column standard UCI setups drop anyway).

### 1.2 Measures

| measure | definition |
|---|---|
| DP_Diff | P(ŷ=1 \| unprivileged) − P(ŷ=1 \| privileged); 0 = parity |
| DIR | ratio of the two selection rates; 1 = parity, <0.8 = adverse impact |
| consistency | k-NN individual-fairness measure from the paper (Eq. 15): per test point, the distance-weighted disagreement between its label and the labels of its 10 nearest neighbours. Lower is better. |

Three settings per dataset, all on the same 80/20 test split (stratified on label and
group):

1. **labels before** – the measures computed from the original labels, before training anything (bias already present in the data)
2. **baseline** – plain logistic regression, no mitigation
3. **LFR** – prototypes fitted on the training fold, predictions (and accuracy/F1) on the test fold

### 1.3 Parameters

| parameter | value | where in code |
|---|---|---|
| random seed | 42 | `src/run_experiment.py`, `src/sweep.py` |
| train/test split | 80/20, stratified on label and group | `src/run_experiment.py` |
| n_prototypes | 10 | `src/run_experiment.py`, `src/sweep.py` |
| max_iter | 100 | `src/run_experiment.py`, `src/sweep.py` |
| fairness_weight | Adult **50.0**, German **5.0**, COMPAS **20.0** | `src/run_experiment.py` |

`max_iter=100` everywhere. I picked it so that all fits converge at fairlearn's default
tolerance (1e-6); they used 21–97 iterations, so the cap never binds.

The `fairness_weight` per dataset in `src/run_experiment.py` comes from the sweep
(`results/sweep.csv`): the lowest weight that reaches |DP_Diff| ≤ 0.06 while keeping DIR
at or above 0.8 and F1 within about 0.02 of the baseline F1.

## 2. Results

All numbers on the same held-out test split. Best fairness value per dataset in **bold**.

### Adult (income > 50K, sex)

| setting | accuracy | F1 | DP_Diff | DIR | consistency |
|---|---|---|---|---|---|
| labels before | – | – | −0.1945 | 0.360 | 0.0078 |
| baseline (logistic regression) | **0.8487** | **0.6566** | −0.1898 | 0.282 | 0.0044 |
| LFR (w=50) | 0.7889 | 0.2934 | **+0.0225** | **1.433** | **0.0013** |

### German Credit (credit granted, age < 25)

| setting | accuracy | F1 | DP_Diff | DIR | consistency |
|---|---|---|---|---|---|
| labels before | – | – | −0.1176 | 0.836 | 0.1727 |
| baseline (logistic regression) | **0.7350** | **0.8191** | −0.1157 | 0.852 | 0.1220 |
| LFR (w=5) | 0.6900 | 0.8133 | **+0.0471** | **1.049** | **0.0205** |

### COMPAS (two-year recidivism, sex)

| setting | accuracy | F1 | DP_Diff | DIR | consistency |
|---|---|---|---|---|---|
| labels before | – | – | −0.1231 | 0.757 | 0.0143 |
| baseline (logistic regression) | **0.6448** | **0.6111** | −0.2614 | 0.458 | 0.0052 |
| LFR (w=20) | 0.5713 | 0.5964 | **−0.0602** | **0.898** | **0.0036** |
