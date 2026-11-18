"""Split conformal prediction and an expanding trust region on ESOL (Delaney 2004).

A gradient boosting model predicts measured log solubility from six simple
molecular descriptors. Split conformal prediction gives 90% intervals. A
"trust region" is defined by kNN distance to the training set: the API answers
only for test points closer than the 80th percentile of calibration distances
and abstains otherwise. Three rounds of simulated new lab data then move the
nearest outside-region test points into training and show the region growing.
"""
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "delaney-processed.csv")
FEATURES = ["Minimum Degree", "Molecular Weight", "Number of H-Bond Donors",
            "Number of Rings", "Number of Rotatable Bonds", "Polar Surface Area"]
TARGET = "measured log solubility in mols per litre"
SEED = 0
ALPHA = 0.10      # 90% target coverage
K = 5             # neighbours for the distance score
TRUST_PCT = 80    # trust threshold percentile of calibration distances
ROUNDS, PER_ROUND = 3, 20


def load():
    if not os.path.exists(DATA):
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    df = pd.read_csv(DATA)
    return df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)


def fit_and_calibrate(X_tr, y_tr, X_cal, y_cal):
    """Fit model, conformal half-width q, and kNN distance scorer with threshold."""
    model = GradientBoostingRegressor(random_state=SEED).fit(X_tr, y_tr)
    scores = np.abs(y_cal - model.predict(X_cal))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    q = np.quantile(scores, level, method="higher")

    scaler = StandardScaler().fit(X_tr)
    nn = NearestNeighbors(n_neighbors=K).fit(scaler.transform(X_tr))

    def dist(X):
        d, _ = nn.kneighbors(scaler.transform(X))
        return d.mean(axis=1)

    threshold = np.percentile(dist(X_cal), TRUST_PCT)
    return model, q, dist, threshold


def summarize(y, pred, q, mask):
    """Coverage, mean width and MAE on the points selected by mask."""
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    err = np.abs(y[mask] - pred[mask])
    return np.mean(err <= q), 2 * q, err.mean()


def main():
    X, y = load()
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=SEED)
    cal, te = train_test_split(rest, test_size=0.5, random_state=SEED)
    print(f"ESOL rows: {len(y)} | train {len(tr)} / calibration {len(cal)} / test {len(te)}")
    print(f"Features: {', '.join(FEATURES)}")

    # Plain split conformal
    model, q, dist, thr = fit_and_calibrate(X[tr], y[tr], X[cal], y[cal])
    pred = model.predict(X[te])
    all_mask = np.ones(len(te), bool)
    cov, width, mae = summarize(y[te], pred, q, all_mask)
    print("\n== Split conformal (target 90%) ==")
    print(f"Conformal half-width q: {q:.3f} log units")
    print(f"Test coverage: {cov:.3f} | mean interval width: {width:.3f} | MAE: {mae:.3f}")

    # Trust region
    d_te = dist(X[te])
    inside = d_te <= thr
    print(f"\n== Trust region (kNN k={K}, threshold = {TRUST_PCT}th pct of calibration distances = {thr:.3f}) ==")
    print(f"{'region':<8}{'n':>5}{'coverage':>10}{'width':>8}{'MAE':>8}")
    for name, m in [("inside", inside), ("outside", ~inside)]:
        c, w, e = summarize(y[te], pred, q, m)
        print(f"{name:<8}{m.sum():>5}{c:>10.3f}{w:>8.3f}{e:>8.3f}")
    print(f"API answers {inside.mean():.3f} of test points, abstains on {1 - inside.mean():.3f}")

    # Expanding trust region with simulated new lab data
    print(f"\n== Expansion: {ROUNDS} rounds, {PER_ROUND} nearest outside points per round moved to training ==")
    print(f"{'round':<6}{'train':>6}{'test':>6}{'answered':>10}{'cov_in':>8}{'cov_out':>9}"
          f"{'cov_all':>9}{'MAE_in':>8}{'MAE_out':>9}{'width':>7}")
    train_idx, test_idx = tr.copy(), te.copy()
    for r in range(ROUNDS + 1):
        if r > 0:
            outside = np.where(~inside)[0]
            take = outside[np.argsort(d_te[outside])[:PER_ROUND]]
            train_idx = np.concatenate([train_idx, test_idx[take]])
            test_idx = np.delete(test_idx, take)
            model, q, dist, thr = fit_and_calibrate(X[train_idx], y[train_idx], X[cal], y[cal])
        pred = model.predict(X[test_idx])
        d_te = dist(X[test_idx])
        inside = d_te <= thr
        yt = y[test_idx]
        c_in, w, e_in = summarize(yt, pred, q, inside)
        c_out, _, e_out = summarize(yt, pred, q, ~inside)
        c_all, _, _ = summarize(yt, pred, q, np.ones(len(yt), bool))
        print(f"{r:<6}{len(train_idx):>6}{len(test_idx):>6}{inside.mean():>10.3f}{c_in:>8.3f}"
              f"{c_out:>9.3f}{c_all:>9.3f}{e_in:>8.3f}{e_out:>9.3f}{w:>7.3f}")


if __name__ == "__main__":
    main()
