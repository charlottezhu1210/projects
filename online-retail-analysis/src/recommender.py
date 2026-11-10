"""User-based collaborative filtering recommender for the Online Retail II dataset.

Evaluation protocol
-------------------
* **Chronological three-way split** by invoice-line volume: train (first 70%), validation (next 15%),
  test (last 15%).
* **Tuning:** the model is fit on *train* only; the neighbourhood size K, the similarity
  sharpness and the own-history blend weight alpha are chosen by mean **NDCG@20** on *validation*.
* **Final evaluation:** the model is refit with the chosen parameters on *train + validation*
  (everything before the test window) and scored on *test* with **NDCG@20** (primary) and
  **Recall@20** (secondary).
* **Baselines:** global popularity (top-20 items by number of distinct buyers) and personal
  purchase history (each user's own most-bought items), scored the same way.
* **Repeat purchases are allowed.** In this wholesale data ~35-45% of the items a customer buys in
  an evaluation window were already bought before, so previously purchased items are neither
  filtered from recommendations nor from the ground truth. The same rule applies to every model
  and every split.

Usage
-----
    python src/recommender.py --data data/online_retail_II.xlsx
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer, OneHotEncoder

NON_PRODUCT_CODES = ["M", "ADJUST", "ADJUST2", "POST"]
TOP_N = 20


# --------------------------------------------------------------------------- data
def load_transactions(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load and clean purchase lines (no returns, no free items, no non-product codes)."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_excel(source)
    df = df.dropna(subset=["Customer ID", "StockCode", "Description"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    df["Amount"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def chronological_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15) -> Split:
    """Split by invoice date so that each part holds the requested share of purchase lines."""
    val_start = df["InvoiceDate"].quantile(1 - val_size - test_size)
    test_start = df["InvoiceDate"].quantile(1 - test_size)
    return Split(
        train=df[df["InvoiceDate"] < val_start],
        val=df[(df["InvoiceDate"] >= val_start) & (df["InvoiceDate"] < test_start)],
        test=df[df["InvoiceDate"] >= test_start],
        val_start=val_start,
        test_start=test_start,
    )


def ground_truth(history: pd.DataFrame, window: pd.DataFrame) -> dict[int, set[str]]:
    """Distinct items each user bought in `window`, restricted to users who appear in `history`."""
    known = set(history["Customer ID"])
    window = window[window["Customer ID"].isin(known)]
    return window.groupby("Customer ID")["StockCode"].agg(set).to_dict()


# ------------------------------------------------------------------------ metrics
def ndcg_at_k(recommended, actual, k: int = TOP_N) -> float:
    actual = set(actual)
    gains = [1 if item in actual else 0 for item in recommended[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(recommended, actual, k: int = TOP_N) -> float:
    actual = set(actual)
    return len(set(recommended[:k]) & actual) / len(actual) if actual else 0.0


def per_user_scores(recommend_fn, truth: dict[int, set[str]], k: int = TOP_N) -> pd.DataFrame:
    """NDCG@k and Recall@k for every user in `truth` (index = Customer ID)."""
    rows = {}
    for user, actual in truth.items():
        recs = recommend_fn(user)
        rows[user] = (ndcg_at_k(recs, actual, k), recall_at_k(recs, actual, k))
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"NDCG@{k}", f"Recall@{k}"])


def evaluate(recommend_fn, truth: dict[int, set[str]], k: int = TOP_N) -> dict[str, float]:
    """Mean NDCG@k and Recall@k over every user in `truth`."""
    scores = per_user_scores(recommend_fn, truth, k)
    return {**scores.mean().to_dict(), "users": len(scores)}


# ---------------------------------------------------------------------- baselines
class PopularityRecommender:
    """Recommend the same top-N items (most distinct buyers in the history window) to everyone."""

    def fit(self, history: pd.DataFrame, top_n: int = TOP_N):
        self.items = history.groupby("StockCode")["Customer ID"].nunique().nlargest(top_n).index.tolist()
        return self

    def recommend(self, user_id) -> list[str]:
        return self.items


class PersonalHistoryRecommender:
    """Recommend each user's own most-purchased items (by number of invoices), padded with popular items."""

    def fit(self, history: pd.DataFrame, top_n: int = TOP_N):
        self.top_n = top_n
        counts = history.groupby(["Customer ID", "StockCode"])["Invoice"].nunique().reset_index(name="n")
        counts = counts.sort_values(["Customer ID", "n"], ascending=[True, False])
        self.user_items = counts.groupby("Customer ID")["StockCode"].apply(list).to_dict()
        self.popular = PopularityRecommender().fit(history, top_n).items
        return self

    def recommend(self, user_id) -> list[str]:
        own = self.user_items.get(user_id, [])[: self.top_n]
        return own + [i for i in self.popular if i not in own][: self.top_n - len(own)]


# ------------------------------------------------------------ collaborative filter
class UserCFRecommender:
    """User-based CF on a blended behaviour (TF-IDF spend) + geography (one-hot country) profile.

    cf(item)    = sum over the K most similar users v of  w(u, v) * spend(v, item) / max spend(v)
                  with w(u, v) = exp(sharpness * cosine(u, v)); sharpness = 0 gives equal weights
    score(item) = (1 - alpha) * cf(item) / max cf  +  alpha * spend(u, item) / max spend(u)

    alpha = 0 is pure collaborative filtering; alpha > 0 blends in the user's own purchase history,
    which matters in wholesale data where many purchases are replenishment.
    """

    def __init__(self, k: int = 30, sharpness: float = 3.0, alpha: float = 0.0,
                 behaviour_weight: float = 0.8, top_n: int = TOP_N):
        self.k, self.sharpness, self.alpha = k, sharpness, alpha
        self.behaviour_weight, self.top_n = behaviour_weight, top_n

    def fit(self, history: pd.DataFrame):
        self.user_item = history.pivot_table(index="Customer ID", columns="StockCode",
                                             values="Amount", aggfunc="sum", fill_value=0)
        self.items = self.user_item.columns.to_numpy()
        self.user_index = {u: i for i, u in enumerate(self.user_item.index)}

        # Behaviour: TF-IDF down-weights items everyone buys, then L2-normalise
        tfidf = TfidfTransformer().fit_transform(csr_matrix(self.user_item.values))
        behaviour = Normalizer().fit_transform(tfidf)

        # Geography: one-hot of each user's (first) country
        country = history.groupby("Customer ID")["Country"].first().reindex(self.user_item.index)
        geography = Normalizer().fit_transform(OneHotEncoder().fit_transform(country.to_frame()))

        w = self.behaviour_weight
        features = hstack([behaviour * w, geography * (1 - w)]).tocsr()
        self.cosine = cosine_similarity(features)
        np.fill_diagonal(self.cosine, -np.inf)            # never pick yourself as a neighbour

        # Each neighbour's spend scaled to [0, 1] so big spenders don't dominate
        values = self.user_item.values
        self.norm_spend = values / np.maximum(values.max(axis=1, keepdims=True), 1e-12)

        # Pre-sort neighbours once; K and sharpness can then be changed without refitting
        self.neighbour_order = np.argsort(-self.cosine, axis=1)
        return self

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def recommend(self, user_id) -> list[str]:
        idx = self.user_index.get(user_id)
        if idx is None:
            return []
        neighbours = self.neighbour_order[idx, : self.k]
        weights = np.exp(self.sharpness * self.cosine[idx, neighbours])
        cf = weights @ self.norm_spend[neighbours]
        scores = (1 - self.alpha) * cf / max(cf.max(), 1e-12) + self.alpha * self.norm_spend[idx]
        top = np.argpartition(-scores, self.top_n)[: self.top_n]
        return self.items[top[np.argsort(-scores[top])]].tolist()


# ----------------------------------------------------------------------- pipeline
def tune(model: UserCFRecommender, truth, k_grid, sharpness_grid, alpha_grid=(0.0,)) -> pd.DataFrame:
    """Grid search on a validation ground truth; rows sorted by NDCG@20 (best first)."""
    rows = []
    for k, s, a in itertools.product(k_grid, sharpness_grid, alpha_grid):
        model.set_params(k=k, sharpness=s, alpha=a)
        rows.append({"K": k, "sharpness": s, "alpha": a, **evaluate(model.recommend, truth)})
    return pd.DataFrame(rows).sort_values(f"NDCG@{TOP_N}", ascending=False, ignore_index=True)


K_GRID = (10, 20, 50, 100, 150, 200, 300)
SHARPNESS_GRID = (0, 1, 3, 5)
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75)


def run_pipeline(source, k_grid=K_GRID, sharpness_grid=SHARPNESS_GRID, alpha_grid=ALPHA_GRID):
    df = load_transactions(source)
    split = chronological_split(df)
    print(f"Train: < {split.val_start:%Y-%m-%d} | validation: < {split.test_start:%Y-%m-%d} | test: rest")

    # 1) Tune on validation with a model fit on train only
    val_truth = ground_truth(split.train, split.val)
    cf = UserCFRecommender().fit(split.train)
    grid = tune(cf, val_truth, k_grid, sharpness_grid, alpha_grid)
    best_cf = grid[grid["alpha"] == 0].iloc[0]          # pure CF
    best_hybrid = grid.iloc[0]                           # CF + own history (alpha tuned too)
    for name, b in [("pure CF", best_cf), ("hybrid", best_hybrid)]:
        print(f"Best {name} on validation: K={int(b.K)}, sharpness={b.sharpness:g}, alpha={b.alpha:g}, "
              f"NDCG@20={b['NDCG@20']:.4f}")

    # 2) Refit on everything before the test window and evaluate once on test
    history = pd.concat([split.train, split.val])
    test_truth = ground_truth(history, split.test)
    models = {
        "Popularity baseline": PopularityRecommender().fit(history),
        "Personal-history baseline": PersonalHistoryRecommender().fit(history),
        "User-based CF (tuned)": UserCFRecommender(k=int(best_cf.K), sharpness=best_cf.sharpness).fit(history),
        "CF + own history (tuned)": UserCFRecommender(k=int(best_hybrid.K), sharpness=best_hybrid.sharpness,
                                                      alpha=best_hybrid.alpha).fit(history),
    }
    results = pd.DataFrame({name: evaluate(m.recommend, test_truth) for name, m in models.items()}).T
    print(results.round(4).to_string())
    return grid, results, models


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=root / "data" / "online_retail_II.xlsx")
    run_pipeline(parser.parse_args().data)
